"""Backtester: replays the three live strategies over historical Alpaca candles.

Run from the project root:
    python -m bot.backtest                      # 6 months of real data (needs keys in .env)
    python -m bot.backtest --synthetic          # engine smoke test on made-up data (NOT results)
    python -m bot.backtest --refresh            # ignore the CSV cache and re-download

What is simulated (it reuses the live strategy / risk / indicator code, so a
backtest exercises the same decisions the bot would take):

* Signals are evaluated on COMPLETED candles and filled at the NEXT candle's
  open - no look-ahead.
* Slippage of 0.05% is applied adversely to every fill (entry, signal exit and
  stop exits). Commission is $0.
* ATR(14) position sizing, hard stop at 1% of equity, ATR trailing stops,
  notional / gross-exposure caps and the post-stop cooldown are the live
  RiskManager rules.
* Stops are checked against each candle's high/low using the levels that were
  in force when the candle opened. If the candle opens through the stop (a gap)
  the fill is at the open, otherwise at the stop price. When both the hard stop
  and the trailing stop are live, the tighter one triggers first.
* The correlation filter (no BTC/USD long while SPY and QQQ are both long) is
  evaluated against the shared portfolio in the combined run.

Not modelled: borrow fees on shorts, partial fills, intrabar path (a stop is
assumed to fill whenever the candle's range touches it), trailing-stop updates
inside a candle (they update at candle close, the live bot updates every 30 s).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from bot import indicators  # noqa: E402
from bot.data_feed import OHLCV, MarketData, bar_seconds  # noqa: E402
from bot.risk_manager import RiskManager, hard_stop_price  # noqa: E402
from bot.strategies import STRATEGY_CLASSES  # noqa: E402

DEFAULT_CAPITAL = 100_000.0
DEFAULT_SLIPPAGE = 0.0005           # 0.05% per fill
DEFAULT_MONTHS = 6
REG_T_MULT = 2.0                    # overnight margin multiple used as the equity buying-power model

# Extra history fetched BEFORE the test window so indicators are warm on day one.
WARMUP_DAYS = {"daily_trend": 420, "mean_reversion": 10, "momentum_breakout": 14, "trend_following": 420}
# Candles handed to a strategy per evaluation (None = everything so far; EMA(200) needs it).
WINDOW_BARS = {"daily_trend": 300, "mean_reversion": 200, "momentum_breakout": 200, "trend_following": None}

STOP_REASONS = ("hard_stop", "trailing_stop")

PARAM_HINTS = {
    "daily_trend": ("STRATEGY_PARAMS['daily_trend']: sma_period (200), stop_atr_mult (3.0 - the hard "
                    "stop distance; position size shrinks as it widens), plus RISK_PER_TRADE_PCT."),
    "mean_reversion": ("STRATEGY_PARAMS['mean_reversion']: std_mult (SPY 1.5 / QQQ 1.8 - raise to "
                       "demand deeper extremes), period (20), plus the exit rule (SMA touch) and "
                       "RISK_PER_TRADE_PCT, which sets how wide the hard stop is."),
    "momentum_breakout": ("STRATEGY_PARAMS['momentum_breakout']: volume_mult (1.5), period (20), "
                          "trailing_atr_mult (2.0), plus RISK_PER_TRADE_PCT."),
    "trend_following": ("STRATEGY_PARAMS['trend_following']: fast_ema / slow_ema (50 / 200), "
                        "trailing_atr_mult (3.0), go_short_on_cross_down, plus RISK_PER_TRADE_PCT."),
}


# --------------------------------------------------------------------------- #
# Simulation state
# --------------------------------------------------------------------------- #
@dataclass
class SimPos:
    symbol: str
    strategy: str
    direction: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    hard_stop_price: float
    trailing_mult: Optional[float]
    extreme_price: float
    trailing_stop: Optional[float]
    last_atr: float


@dataclass
class SimTrade:
    symbol: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str
    hard_stop: float

    @property
    def exit_kind(self) -> str:
        for kind in ("hard_stop", "trailing_stop", "end_of_test"):
            if self.reason.startswith(kind):
                return kind
        return "signal"


@dataclass
class _Inst:
    inst: object
    strat: object
    df: pd.DataFrame
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    first_idx: int
    window: Optional[int]
    dur: pd.Timedelta
    cooldown_until: int = -1


@dataclass
class RunResult:
    name: str
    symbols: List[str]
    strategies: List[str]
    trades: List[SimTrade]
    curve: pd.Series
    stats: dict
    blocked_by_filter: int = 0
    skipped: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Backtester:
    def __init__(self, data: Dict[str, pd.DataFrame], start_ts: pd.Timestamp,
                 symbols: Sequence[str], capital: float = DEFAULT_CAPITAL,
                 slippage: float = DEFAULT_SLIPPAGE, correlation_filter: bool = True,
                 name: str = ""):
        self.name = name or "+".join(symbols)
        self.symbols = list(symbols)
        self.capital = float(capital)
        self.slip = float(slippage)
        self.correlation_filter = correlation_filter
        self.risk = RiskManager()
        self.positions: Dict[str, SimPos] = {}
        self.pending: Dict[str, dict] = {}
        self.trades: List[SimTrade] = []
        self.last_px: Dict[str, float] = {}
        self.realized = 0.0
        self.blocked = 0
        self.skipped: Counter = Counter()      # entry attempts that did not become trades, by reason
        self.curve: List[tuple] = []
        self.inst: Dict[str, _Inst] = {}
        self.start_ts = start_ts

        for sym in self.symbols:
            inst = config.INSTRUMENTS[sym]
            strat = STRATEGY_CLASSES[inst.strategy](config.STRATEGY_PARAMS[inst.strategy])
            df = data[sym]
            self.inst[sym] = _Inst(
                inst=inst, strat=strat, df=df,
                o=df["open"].to_numpy(), h=df["high"].to_numpy(),
                l=df["low"].to_numpy(), c=df["close"].to_numpy(),
                atr=indicators.atr(df, config.ATR_PERIOD).to_numpy(),
                first_idx=int(df.index.searchsorted(start_ts)),
                window=WINDOW_BARS[inst.strategy],
                dur=pd.Timedelta(seconds=bar_seconds(strat.timeframe)),
            )

    # ------------------------------------------------------------ accounting
    def _unrealized(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            sign = 1 if pos.direction == "long" else -1
            total += sign * (self.last_px.get(sym, pos.entry_price) - pos.entry_price) * pos.qty
        return total

    def _equity(self) -> float:
        return self.capital + self.realized + self._unrealized()

    def _gross(self) -> float:
        return sum(abs(p.qty * self.last_px.get(s, p.entry_price)) for s, p in self.positions.items())

    # ----------------------------------------------------------------- orders
    def _close(self, st: _Inst, j: int, raw_price: float, reason: str,
               when: pd.Timestamp) -> None:
        sym = st.inst.symbol
        pos = self.positions.pop(sym)
        fill = raw_price * (1 - self.slip) if pos.direction == "long" else raw_price * (1 + self.slip)
        sign = 1 if pos.direction == "long" else -1
        pnl = sign * (fill - pos.entry_price) * pos.qty
        self.realized += pnl
        self.trades.append(SimTrade(sym, pos.strategy, pos.direction, pos.entry_time, when,
                                    pos.entry_price, fill, pos.qty, pnl, reason,
                                    pos.hard_stop_price))
        self.last_px[sym] = fill
        if reason.startswith(STOP_REASONS):
            st.cooldown_until = j + config.COOLDOWN_BARS_AFTER_STOP + 1

    def _open(self, st: _Inst, j: int, direction: str, atr: float) -> None:
        inst = st.inst
        sym = inst.symbol
        if direction == "short" and not inst.allow_short:
            self.skipped["short_not_allowed"] += 1
            return
        if j < st.cooldown_until:
            self.skipped["cooldown"] += 1
            return
        if self.correlation_filter:
            allowed, _ = self.risk.correlation_allows(sym, direction, self.positions)
            if not allowed:
                self.blocked += 1
                self.skipped["correlation_filter"] += 1
                return
        if not (math.isfinite(atr) and atr > 0):
            self.skipped["atr_unavailable"] += 1
            return
        open_px = float(st.o[j])
        equity, gross = self._equity(), self._gross()
        if inst.asset_class == config.CRYPTO:
            bp = max(0.0, equity - gross)                  # crypto is cash-only
        else:
            bp = max(0.0, equity * REG_T_MULT - gross)
        size = self.risk.position_size(inst, open_px, atr, equity, bp, gross,
                                       getattr(st.strat, "stop_atr_mult", 1.0))
        if size is None:
            no_room = equity * config.MAX_GROSS_EXPOSURE_PCT - gross <= 0 or bp <= 0
            self.skipped["exposure_cap_full" if no_room else "size_too_small"] += 1
            return
        fill = open_px * (1 + self.slip) if direction == "long" else open_px * (1 - self.slip)
        mult = st.strat.trailing_atr_mult
        trailing = None
        if mult:
            trailing = fill - mult * atr if direction == "long" else fill + mult * atr
        self.positions[sym] = SimPos(
            symbol=sym, strategy=inst.strategy, direction=direction, qty=size.qty,
            entry_price=fill, entry_time=st.df.index[j],
            hard_stop_price=hard_stop_price(direction, fill, size.qty, size.risk_dollars),
            trailing_mult=mult, extreme_price=fill, trailing_stop=trailing, last_atr=atr,
        )
        self.last_px[sym] = fill

    def _execute_pending(self, st: _Inst, j: int, pend: dict) -> None:
        sym = st.inst.symbol
        when = st.df.index[j]
        if pend["exit"] and sym in self.positions:
            self._close(st, j, float(st.o[j]), pend["reason"], when)
        if pend["enter"] and sym not in self.positions:
            self._open(st, j, pend["enter"], pend["atr"])

    def _check_stops(self, st: _Inst, j: int) -> None:
        sym = st.inst.symbol
        pos = self.positions.get(sym)
        if pos is None:
            return
        o, h, l = float(st.o[j]), float(st.h[j]), float(st.l[j])
        levels = [("hard_stop", pos.hard_stop_price)]
        if pos.trailing_stop is not None:
            levels.append(("trailing_stop", pos.trailing_stop))
        when = st.df.index[j]
        if pos.direction == "long":
            kind, level = max(levels, key=lambda kv: kv[1])        # tighter = higher
            if l <= level:
                self._close(st, j, min(o, level), kind, when)
        else:
            kind, level = min(levels, key=lambda kv: kv[1])        # tighter = lower
            if h >= level:
                self._close(st, j, max(o, level), kind, when)

    def _on_bar_close(self, st: _Inst, j: int) -> None:
        sym = st.inst.symbol
        self.last_px[sym] = float(st.c[j])
        pos = self.positions.get(sym)
        atr_now = float(st.atr[j])
        if pos and pos.trailing_mult:
            extreme = float(st.h[j]) if pos.direction == "long" else float(st.l[j])
            self.risk.update_trailing_stop(pos, extreme, atr_now)
        lo = 0 if st.window is None else max(0, j + 1 - st.window)
        sig = st.strat.evaluate(sym, st.df.iloc[lo:j + 1], pos.direction if pos else None)
        if not sig.is_hold:
            self.pending[sym] = {"exit": sig.exit, "enter": sig.enter,
                                 "reason": f"signal: {sig.reason}", "atr": atr_now}

    # -------------------------------------------------------------------- run
    def run(self) -> RunResult:
        events = []
        for order, sym in enumerate(self.symbols):
            st = self.inst[sym]
            idx = st.df.index
            for j in range(st.first_idx, len(idx)):
                events.append(((idx[j] + st.dur).value, order, sym, j))
        events.sort()
        self.curve = [(self.start_ts, self.capital)]

        for end_ns, _, sym, j in events:
            st = self.inst[sym]
            pend = self.pending.pop(sym, None)
            if pend:
                self._execute_pending(st, j, pend)
            self._check_stops(st, j)
            self._on_bar_close(st, j)
            self.curve.append((pd.Timestamp(end_ns, tz="UTC"), self._equity()))

        # Mark the end of the test: flatten whatever is still open at the last close.
        for sym in list(self.positions):
            st = self.inst[sym]
            self._close(st, len(st.df) - 1, float(st.c[-1]), "end_of_test",
                        st.df.index[-1] + st.dur)
        if events:
            self.curve.append((pd.Timestamp(events[-1][0], tz="UTC"), self._equity()))

        curve = pd.Series([v for _, v in self.curve],
                          index=pd.DatetimeIndex([t for t, _ in self.curve]), name=self.name)
        curve = curve[~curve.index.duplicated(keep="last")]
        calendar = any(self.inst[s].inst.asset_class == config.CRYPTO for s in self.symbols)
        first_px = {s: float(self.inst[s].o[self.inst[s].first_idx]) for s in self.symbols
                    if self.inst[s].first_idx < len(self.inst[s].df)}
        stats = compute_stats(self.trades, curve, self.capital, calendar)
        if len(self.symbols) == 1 and self.symbols[0] in first_px:
            s = self.symbols[0]
            stats["buy_hold_pct"] = (float(self.inst[s].c[-1]) / first_px[s] - 1) * 100
        return RunResult(self.name, self.symbols,
                         sorted({self.inst[s].inst.strategy for s in self.symbols}),
                         self.trades, curve, stats, self.blocked, dict(self.skipped))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def sharpe_ratio(curve: pd.Series, capital: float, calendar_days: bool) -> float:
    """Annualised Sharpe of daily equity returns (risk-free rate 0).

    calendar_days=False: only days that have candles (equities, sqrt(252)).
    calendar_days=True : every calendar day, weekends carried flat (anything
    including 24/7 crypto, sqrt(365)).
    """
    if len(curve) < 2:
        return float("nan")
    day_key = curve.index.tz_convert(config.MARKET_TZ).date
    daily = curve.groupby(day_key).last()
    daily.index = pd.DatetimeIndex(daily.index)
    if calendar_days:
        daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1], freq="D")).ffill()
    daily = pd.concat([pd.Series([capital], index=[daily.index[0] - pd.Timedelta(days=1)]), daily])
    rets = daily.pct_change().dropna()
    if len(rets) < 2:
        return float("nan")
    sd = rets.std(ddof=1)
    if not sd or not math.isfinite(sd):
        return float("nan")
    return float(rets.mean() / sd * math.sqrt(365 if calendar_days else 252))


def compute_stats(trades: List[SimTrade], curve: pd.Series, capital: float,
                  calendar_days: bool) -> dict:
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    n = len(pnls)
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = float("inf") if gross_win > 0 else float("nan")
    peak = curve.cummax()
    dd = (curve / peak - 1.0) * 100
    return {
        "trades": n,
        "win_rate": (len(wins) / n * 100) if n else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,   # negative number
        "profit_factor": pf,
        "net_pnl": float(pnls.sum()),
        "max_dd_pct": float(dd.min()) if len(dd) else 0.0,
        "max_dd_usd": float((curve - peak).min()) if len(curve) else 0.0,
        "sharpe": sharpe_ratio(curve, capital, calendar_days),
        "return_pct": (float(curve.iloc[-1]) / capital - 1) * 100,
        "exit_mix": dict(Counter(t.exit_kind for t in trades)),
        "longs": sum(t.direction == "long" for t in trades),
        "shorts": sum(t.direction == "short" for t in trades),
    }


# --------------------------------------------------------------------------- #
# Suite: per instrument, per strategy, combined (filter on / off)
# --------------------------------------------------------------------------- #
def run_suite(data: Dict[str, pd.DataFrame], start_ts: pd.Timestamp, capital: float,
              slippage: float) -> Dict[str, RunResult]:
    out: Dict[str, RunResult] = {}
    for sym in config.INSTRUMENTS:
        out[sym] = Backtester(data, start_ts, [sym], capital, slippage, name=sym).run()
    for strat in config.active_strategies():
        syms = [s for s, i in config.INSTRUMENTS.items() if i.strategy == strat]
        out[f"strategy:{strat}"] = Backtester(data, start_ts, syms, capital, slippage,
                                              name=f"strategy:{strat}").run()
    allsyms = list(config.INSTRUMENTS)
    out["combined"] = Backtester(data, start_ts, allsyms, capital, slippage,
                                 correlation_filter=True, name="combined").run()
    out["combined_nofilter"] = Backtester(data, start_ts, allsyms, capital, slippage,
                                          correlation_filter=False,
                                          name="combined_nofilter").run()
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _f(x, spec, na="n/a"):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return na
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return format(x, spec)


def print_summary(results: Dict[str, RunResult], start_ts, end_ts, capital: float,
                  slippage: float, synthetic: bool) -> List[str]:
    """Print the summary table; return the names of rows with a negative Sharpe."""
    line = "=" * 128
    print("\n" + line)
    if synthetic:
        print("*** SYNTHETIC DATA - this only proves the engine runs. These are NOT real results. ***")
    print(f"BACKTEST SUMMARY | {start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d} | start equity ${capital:,.0f} per run | "
          f"slippage {slippage * 100:.2f}%/fill | commission $0")
    print(line)
    header = (f"{'':<34}{'Trades':>7}{'Win%':>7}{'AvgWin$':>10}{'AvgLoss$':>10}{'ProfitF':>9}"
              f"{'MaxDD%':>9}{'Sharpe':>8}{'Return%':>9}{'B&H%':>8}{'NetP&L$':>11}")

    def row(label, r: RunResult):
        s = r.stats
        label = label + ("*" if s["trades"] < 10 else "")
        return (f"{label:<34}{s['trades']:>7d}{_f(s['win_rate'], '.1f'):>7}{_f(s['avg_win'], ',.0f'):>10}"
                f"{_f(s['avg_loss'], ',.0f'):>10}{_f(s['profit_factor'], '.2f'):>9}"
                f"{_f(s['max_dd_pct'], '.2f'):>9}{_f(s['sharpe'], '.2f'):>8}{_f(s['return_pct'], '+.2f'):>9}"
                f"{_f(s.get('buy_hold_pct'), '+.1f'):>8}{_f(s['net_pnl'], '+,.0f'):>11}")

    flagged: List[str] = []

    def section(title, keys, label_fn):
        print(f"\n{title}")
        print(header)
        print("-" * len(header))
        for k in keys:
            r = results[k]
            print(row(label_fn(k, r), r))
            sh = r.stats["sharpe"]
            if isinstance(sh, float) and math.isfinite(sh) and sh < 0:
                flagged.append(k)

    section("PER INSTRUMENT (standalone, own account)", list(config.INSTRUMENTS),
            lambda k, r: f"{k} [{config.INSTRUMENTS[k].strategy.replace('_', ' ')}]")
    section("PER STRATEGY (its instruments share one account)",
            [f"strategy:{s}" for s in config.active_strategies()],
            lambda k, r: k.split(":")[1].replace("_", " ") + f" ({'+'.join(r.symbols)})")
    section("COMBINED PORTFOLIO (all 5 instruments, shared account)", ["combined", "combined_nofilter"],
            lambda k, r: "COMBINED, corr. filter ON" if k == "combined" else "COMBINED, filter OFF")

    c, n = results["combined"], results["combined_nofilter"]
    if {rule["blocked"] for rule in config.CORRELATION_RULES} & set(config.INSTRUMENTS):
        print(f"\nCorrelation filter: blocked {c.blocked_by_filter} BTC/USD long entr"
              f"{'y' if c.blocked_by_filter == 1 else 'ies'}; "
              f"return {c.stats['return_pct']:+.2f}% with the filter vs {n.stats['return_pct']:+.2f}% without; "
              f"max drawdown {c.stats['max_dd_pct']:.2f}% vs {n.stats['max_dd_pct']:.2f}%.")
    else:
        print("\nCorrelation filter: inactive (no instrument it blocks is enabled).")

    for label, r in (("filter ON ", c), ("filter OFF", n)):
        sk = r.skipped
        txt = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(sk.items())) or "none"
        print(f"Entries skipped, combined {label}: {txt}")
    if c.skipped.get("correlation_filter", 0) and c.stats["trades"] == n.stats["trades"]:
        print("  -> Filter ON and OFF produced the same trades: every BTC/USD long the filter blocked would "
              "also have been refused by the gross-exposure cap (SPY + QQQ at their per-position caps already use "
              f"{config.MAX_GROSS_EXPOSURE_PCT * 100:.0f}% of equity).")

    print("\nEXIT MIX (how trades ended)")
    for k, r in results.items():
        mix = r.stats["exit_mix"]
        tot = max(1, r.stats["trades"])
        parts = ", ".join(f"{name} {mix.get(name, 0)} ({mix.get(name, 0) / tot * 100:.0f}%)"
                          for name in ("signal", "hard_stop", "trailing_stop", "end_of_test"))
        print(f"  {k:<34}{parts}")

    print("\n* fewer than 10 trades: the statistics for that row are not meaningful.")
    print("Sharpe: annualised from daily equity returns, risk-free 0 (sqrt(252) on trading days for "
          "equity-only rows; sqrt(365) on calendar days for rows containing BTC/USD).")
    print("Max drawdown: largest peak-to-trough fall of the candle-by-candle equity curve. "
          "Profit factor: gross profit / gross loss. B&H: buy-and-hold of the instrument over the window.")

    print("\n" + line)
    if flagged:
        print("FLAGS - NEGATIVE SHARPE OVER THE TEST WINDOW")
        print(line)
        for k in flagged:
            r = results[k]
            s = r.stats
            mix = s["exit_mix"]
            tot = max(1, s["trades"])
            stop_share = (mix.get("hard_stop", 0) + mix.get("trailing_stop", 0)) / tot * 100
            payoff = (s["avg_win"] / abs(s["avg_loss"])) if s["avg_loss"] else float("nan")
            low = "  (only %d trades - low confidence)" % s["trades"] if s["trades"] < 10 else ""
            print(f"[!] {k}: Sharpe {s['sharpe']:.2f}, return {s['return_pct']:+.2f}%, "
                  f"win rate {_f(s['win_rate'], '.0f')}%, payoff {_f(payoff, '.2f')}x, "
                  f"{stop_share:.0f}% of exits were stops{low}")
            for strat in r.strategies:
                print(f"      adjust -> {PARAM_HINTS[strat]}")
        print("\nNote: a negative Sharpe over a single 6-month window is evidence, not proof - "
              "check the trade count and the B&H column before retuning, and avoid fitting to this one window.")
    else:
        print("No row has a negative Sharpe ratio over this window.")
    print(line)
    return flagged


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
# Light-mode tokens from the validated reference palette (blue, orange, aqua,
# yellow, magenta pass the adjacent-pair CVD / normal-vision gates; three of
# them sit under 3:1 on the surface, so every line also gets a direct label
# and there is a legend plus the printed summary table).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _daily(curve: pd.Series) -> pd.Series:
    key = curve.index.tz_convert(config.MARKET_TZ).date
    d = curve.groupby(key).last()
    d.index = pd.DatetimeIndex(d.index)
    return d.reindex(pd.date_range(d.index[0], d.index[-1], freq="D")).ffill()


def _spread_labels(ys: List[float], min_gap: float) -> List[float]:
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def plot_results(results: Dict[str, RunResult], path: Path, capital: float,
                 synthetic: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.6]}, facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Panel 1 - combined portfolio, filter on vs off
    on, off = _daily(results["combined"].curve), _daily(results["combined_nofilter"].curve)
    ax = axes[0]
    ax.plot(off.index, off.values, color=INK2, linewidth=1.4, linestyle="--",
            label="Combined, correlation filter OFF")
    ax.plot(on.index, on.values, color=INK, linewidth=2.0, label="Combined, correlation filter ON")
    ax.axhline(capital, color=GRID, linewidth=1.0)
    if len(on) == len(off) and np.allclose(on.values, off.values):
        ax.text(0.99, 0.04, "Filter ON and OFF curves are identical: the exposure cap already blocked those entries",
                transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    ax.set_ylabel("Portfolio equity ($)")
    ax.set_title("Combined portfolio equity", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Panel 2 - each instrument standalone, % return
    ax = axes[1]
    ends = []
    for k, sym in enumerate(config.INSTRUMENTS):
        d = _daily(results[sym].curve)
        pct = (d / capital - 1) * 100
        ax.plot(pct.index, pct.values, color=SERIES[k], linewidth=1.6,
                label=f"{sym} ({config.INSTRUMENTS[sym].strategy.replace('_', ' ')})")
        ends.append((pct.index[-1], float(pct.iloc[-1]), sym))
    span = max(abs(y) for _, y, _ in ends) or 1.0
    ys = _spread_labels([y for _, y, _ in ends], span * 0.07)
    for (x, y, sym), y_lab in zip(ends, ys):
        ax.annotate(f"{sym} {y:+.1f}%", xy=(x, y), xytext=(x + pd.Timedelta(days=2), y_lab),
                    color=INK, fontsize=9, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.set_ylabel("Return vs start (%)")
    ax.set_title("Each instrument standalone ($100k own account)", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    # Panel 3 - combined drawdown
    ax = axes[2]
    dd = (on / on.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, color=INK2, alpha=0.25, linewidth=0)
    ax.plot(dd.index, dd.values, color=INK2, linewidth=1.4)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Combined portfolio drawdown (filter ON)", loc="left", fontsize=12, fontweight="bold")

    c = results["combined"].stats
    sub = (f"Combined (filter on): return {c['return_pct']:+.2f}%  |  max drawdown {c['max_dd_pct']:.2f}%  |  "
           f"Sharpe {_f(c['sharpe'], '.2f')}  |  {c['trades']} trades")
    fig.suptitle(("SYNTHETIC DATA - NOT REAL RESULTS\n" if synthetic else "") + sub,
                 x=0.01, ha="left", fontsize=11, color="#b3261e" if synthetic else INK)
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data: real (Alpaca + CSV cache) and synthetic (engine smoke test)
# --------------------------------------------------------------------------- #
def _cache_file(cache_dir: Path, sym: str, spec) -> Path:
    return cache_dir / f"{sym.replace('/', '')}_{spec[0]}{spec[1]}.csv"


def load_history(md: MarketData, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                 cache_dir: Path, refresh: bool) -> Dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, pd.DataFrame] = {}
    for sym, inst in config.INSTRUMENTS.items():
        params = config.STRATEGY_PARAMS[inst.strategy]
        spec = tuple(params["timeframe"])
        fetch_from = start_ts - pd.Timedelta(days=WARMUP_DAYS[inst.strategy])
        path = _cache_file(cache_dir, sym, spec)
        df = None
        if path.exists() and not refresh:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index, utc=True)
            if (not cached.empty and cached.index[0] <= fetch_from + pd.Timedelta(days=7)
                    and cached.index[-1] >= end_ts - pd.Timedelta(days=4)):
                df = cached
                print(f"  {sym:<8} cache hit  {path.name}")
            else:
                print(f"  {sym:<8} cache does not cover the window; re-downloading")
        if df is None:
            print(f"  {sym:<8} downloading {spec[0]}{spec[1]} bars from {fetch_from:%Y-%m-%d} ...")
            df = md.get_bars_between(inst, spec, fetch_from.to_pydatetime(), end_ts.to_pydatetime())
            if df.empty:
                raise SystemExit(f"No bars returned for {sym}. Check your keys / data subscription "
                                 f"(ALPACA_DATA_FEED={config.DATA_FEED}).")
            df.to_csv(path)
        data[sym] = df
    return data


def make_synthetic_data(start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                        seed: int = 7) -> Dict[str, pd.DataFrame]:
    """Made-up candles with the right shapes (sessions, timeframes, volume) so the
    engine can be exercised offline. They carry NO information about real markets."""
    rng = np.random.default_rng(seed)
    first = start_ts - pd.Timedelta(days=WARMUP_DAYS["trend_following"] + 10)
    days = pd.bdate_range(first.tz_convert(None).normalize(), end_ts.tz_convert(None).normalize())

    def session(times):
        parts = [pd.DatetimeIndex([pd.Timestamp(d).tz_localize(config.MARKET_TZ) + t
                                   for t in times]) for d in days]
        return parts[0].append(parts[1:]).tz_convert("UTC")

    idx15 = session([pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * i) for i in range(26)])
    idx4h = session([pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=13, minutes=30)])
    idx1h = pd.date_range(start_ts - pd.Timedelta(days=40), end_ts, freq="1h", tz="UTC")

    def ohlc(close, base, vol, sigma):
        close = np.asarray(close)
        open_ = np.concatenate([[close[0]], close[:-1]])
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma * 0.5, len(close))))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma * 0.5, len(close))))
        return open_, hi, lo, close

    def ou(n, sigma, theta, noise):
        level = np.cumsum(rng.normal(0, sigma * 0.15, n))
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = x[t - 1] + theta * (level[t] - x[t - 1]) + sigma * noise[t]
        return x

    def regime_walk(n, drift, sigma, flip):
        sign, out = 1.0, np.zeros(n)
        for t in range(1, n):
            if rng.random() < flip:
                sign = -sign
            out[t] = out[t - 1] + sign * drift + sigma * rng.standard_normal()
        return out

    def frame(idx, o, h, l, c, v):
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)

    data = {}
    n = len(idx15)
    common = rng.standard_normal(n)
    for sym, base, sigma in (("SPY", 700.0, 0.0007), ("QQQ", 600.0, 0.0009)):
        noise = 0.6 * common + 0.8 * rng.standard_normal(n)
        c = base * np.exp(ou(n, sigma, 0.04, noise))
        o, h, l, c = ohlc(c, base, 0, sigma)
        data[sym] = frame(idx15, o, h, l, c, rng.lognormal(11, 0.4, n))

    n = len(idx1h)
    sigma = 0.004
    r = regime_walk(n, 0.0003, sigma, 1 / 200) + 0 * rng.standard_normal(n)
    c = 85000 * np.exp(r)
    o, h, l, c = ohlc(c, 85000, 0, sigma)
    vol = 0.05 * rng.lognormal(0, 0.6, n) * (1 + 4 * np.abs(np.diff(np.log(c), prepend=np.log(c[0]))) / sigma)
    if "BTC/USD" in config.INSTRUMENTS:
        data["BTC/USD"] = frame(idx1h, o, h, l, c, vol)

    n = len(idx4h)
    for sym, base, sigma, drift in (("GLD", 400.0, 0.006, 0.0012), ("USO", 80.0, 0.01, 0.0015)):
        c = base * np.exp(regime_walk(n, drift, sigma, 1 / 90))
        o, h, l, c = ohlc(c, base, 0, sigma)
        data[sym] = frame(idx4h, o, h, l, c, rng.lognormal(12, 0.3, n))

    # Instruments on the daily strategy get one bar per business day (Alpaca stamps
    # daily bars at 04:00/05:00 UTC), replacing the intraday series built above.
    idxd = pd.DatetimeIndex(days).tz_localize("UTC") + pd.Timedelta(hours=4)
    for sym, base, sigma, drift in (("SPY", 700.0, 0.010, 0.0006), ("QQQ", 600.0, 0.013, 0.0008),
                                    ("GLD", 400.0, 0.008, 0.0004)):
        if config.INSTRUMENTS[sym].strategy != "daily_trend":
            continue
        n = len(idxd)
        c = base * np.exp(regime_walk(n, drift, sigma, 1 / 120))
        o, h, l, c = ohlc(c, base, 0, sigma)
        data[sym] = frame(idxd, o, h, l, c, rng.lognormal(15, 0.3, n))
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backtest the 3 strategies on 5 instruments.")
    ap.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE,
                    help="fraction per fill, 0.0005 = 0.05%%")
    ap.add_argument("--end", help="last day of the window, YYYY-MM-DD (default: now)")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the CSV cache")
    ap.add_argument("--synthetic", action="store_true", help="use made-up data (engine smoke test only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache-dir", default=str(ROOT / "backtest_data"))
    ap.add_argument("--output", default=None, help="chart path (default backtest_results.png)")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    end_ts = (pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) if args.end
              else pd.Timestamp(datetime.now(timezone.utc)))
    start_ts = end_ts - pd.DateOffset(months=args.months)

    if args.synthetic:
        print("SYNTHETIC mode: generating made-up candles (not real market data).")
        data = make_synthetic_data(start_ts, end_ts, args.seed)
    else:
        if not config.API_KEY or not config.API_SECRET or "your_" in config.API_KEY.lower():
            print("Set APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env to download data "
                  "(or use --synthetic for an engine smoke test).", file=sys.stderr)
            return 2
        from alpaca_trade_api.rest import REST
        api = REST(config.API_KEY, config.API_SECRET, config.BASE_URL, api_version="v2")
        print(f"Loading {args.months} months of history (+ indicator warm-up) ...")
        data = load_history(MarketData(api), start_ts, end_ts, Path(args.cache_dir), args.refresh)

    print("\nData coverage in the test window:")
    for sym, df in data.items():
        i0 = int(df.index.searchsorted(start_ts))
        spec = config.STRATEGY_PARAMS[config.INSTRUMENTS[sym].strategy]["timeframe"]
        print(f"  {sym:<8} {spec[0]}{spec[1]:<6} {len(df) - i0:>6} bars in window, {i0:>6} warm-up bars "
              f"| {df.index[i0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        need = config.STRATEGY_PARAMS["trend_following"]["min_bars"]
        if config.INSTRUMENTS[sym].strategy == "trend_following" and i0 < need:
            print(f"    WARNING: only {i0} warm-up bars (< {need}); the 200 EMA will not be "
                  f"trusted for the early part of the window.")

    results = run_suite(data, start_ts, args.capital, args.slippage)
    print_summary(results, start_ts, end_ts, args.capital, args.slippage, args.synthetic)

    trades_path = ROOT / ("backtest_trades_synthetic.csv" if args.synthetic else "backtest_trades.csv")
    pd.DataFrame([{
        "symbol": t.symbol, "strategy": t.strategy, "direction": t.direction,
        "entry_time": t.entry_time, "exit_time": t.exit_time, "entry_price": round(t.entry_price, 4),
        "exit_price": round(t.exit_price, 4), "qty": t.qty, "pnl": round(t.pnl, 2),
        "exit_reason": t.reason, "hard_stop": t.hard_stop,
    } for t in results["combined"].trades]).to_csv(trades_path, index=False)
    print(f"Combined-portfolio trades written to {trades_path.name}")

    if not args.no_chart:
        out = Path(args.output) if args.output else ROOT / (
            "backtest_results_synthetic.png" if args.synthetic else "backtest_results.png")
        plot_results(results, out, args.capital, args.synthetic)
        print(f"Equity-curve chart saved to {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

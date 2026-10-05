"""Backtest engine tests: fills, slippage, stops, filter, metrics. All offline."""
import math

import numpy as np
import pandas as pd
import pytest

import config
from bot import backtest as bt
from bot.backtest import Backtester, SimPos, SimTrade, compute_stats, sharpe_ratio
from bot.strategies.base import Signal

SLIP = 0.0005
pytestmark = pytest.mark.usefixtures("legacy_instruments")


def make_df(n=60, freq="15min", start="2026-03-02 14:30", price=100.0, rng=1.0):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    c = np.full(n, price)
    return pd.DataFrame({"open": c, "high": c + rng, "low": c - rng, "close": c,
                         "volume": np.full(n, 1000.0)}, index=idx)


def make_bt(symbols=("SPY",), start_idx=20, **kw):
    data = {}
    for s in symbols:
        spec = config.STRATEGY_PARAMS[config.INSTRUMENTS[s].strategy]["timeframe"]
        freq = "15min" if spec[1] == "Minute" else f"{spec[0]}h"
        data[s] = make_df(freq=freq)
    start_ts = data[symbols[0]].index[start_idx]
    b = Backtester(data, start_ts, list(symbols), 100_000.0, SLIP, **kw)
    for st in b.inst.values():                      # writable copies so tests can script bars
        st.o, st.h, st.l, st.c = (a.copy() for a in (st.o, st.h, st.l, st.c))
    return b


class Stub:
    """Strategy stub: emits a scripted signal on a given bar count."""
    timeframe = (15, "Minute")
    trailing_atr_mult = None

    def __init__(self, script):
        self.script = script     # {len(bars): Signal}

    def evaluate(self, symbol, bars, direction):
        return self.script.get(len(bars), Signal(False, None, "hold", {}))


def test_entry_fills_at_open_with_adverse_slippage_and_exact_1pct_stop():
    b = make_bt()
    st = b.inst["SPY"]
    b._open(st, 21, "long", atr=2.0)
    pos = b.positions["SPY"]
    assert pos.entry_price == pytest.approx(100 * (1 + SLIP))
    assert pos.qty == pytest.approx(500)                  # 1% of 100k / ATR 2
    loss_at_stop = (pos.entry_price - pos.hard_stop_price) * pos.qty
    assert loss_at_stop == pytest.approx(1000.0)          # exactly 1% of equity
    b2 = make_bt()
    b2._open(b2.inst["SPY"], 21, "short", atr=2.0)
    assert b2.positions["SPY"].entry_price == pytest.approx(100 * (1 - SLIP))


def test_signal_is_filled_at_next_bar_open_not_signal_bar():
    b = make_bt()
    st = b.inst["SPY"]
    first = st.first_idx
    # make next bar's open distinct from the signal bar's close
    st.o[first + 1] = 105.0
    st.strat = Stub({first + 1 + 0: Signal(False, "long", "test", {})})
    # evaluate() receives iloc[lo:j+1]; the window for SPY is 200 bars -> len == j+1
    b.run()
    t = b.trades[0]
    assert t.entry_time == st.df.index[first + 1]
    assert t.entry_price == pytest.approx(105.0 * (1 + SLIP))   # next open, not close (100)


def test_hard_stop_fills_at_stop_level_when_no_gap():
    b = make_bt()
    st = b.inst["SPY"]
    b._open(st, 21, "long", atr=2.0)
    stop = b.positions["SPY"].hard_stop_price
    st.o[22], st.l[22] = 100.0, stop - 1.0               # trades through the stop intrabar
    b._check_stops(st, 22)
    t = b.trades[0]
    assert t.exit_kind == "hard_stop"
    assert t.exit_price == pytest.approx(stop * (1 - SLIP))


def test_hard_stop_gap_fills_at_open_not_stop_level():
    b = make_bt()
    st = b.inst["SPY"]
    b._open(st, 21, "long", atr=2.0)
    stop = b.positions["SPY"].hard_stop_price
    st.o[22], st.l[22] = stop - 5.0, stop - 6.0          # gaps below the stop
    b._check_stops(st, 22)
    t = b.trades[0]
    assert t.exit_price == pytest.approx((stop - 5.0) * (1 - SLIP))
    assert t.pnl < -1000.0                               # gap loss exceeds the 1% budget
    assert st.cooldown_until > 22                        # cooldown after a stop


def test_short_stop_triggers_on_high():
    b = make_bt()
    st = b.inst["SPY"]
    b._open(st, 21, "short", atr=2.0)
    stop = b.positions["SPY"].hard_stop_price
    assert stop > b.positions["SPY"].entry_price
    st.o[22], st.h[22] = 100.0, stop + 0.5
    b._check_stops(st, 22)
    assert b.trades[0].exit_price == pytest.approx(stop * (1 + SLIP))


def test_open_position_is_closed_at_end_of_test():
    b = make_bt()
    st = b.inst["SPY"]
    st.strat = Stub({st.first_idx + 1: Signal(False, "long", "test", {})})
    res = b.run()
    assert not b.positions
    assert len(res.trades) == 1 and res.trades[0].exit_kind == "end_of_test"
    assert res.trades[0].exit_price == pytest.approx(float(st.c[-1]) * (1 - SLIP))


def test_correlation_filter_blocks_btc_long_when_spy_and_qqq_long():
    def seeded(filter_on):
        b = make_bt(("SPY", "QQQ", "BTC/USD"), correlation_filter=filter_on)
        for s in ("SPY", "QQQ"):
            b.positions[s] = SimPos(s, "mean_reversion", "long", 10, 100.0,
                                    b.inst[s].df.index[0], 90.0, None, 100.0, None, 1.0)
        b._open(b.inst["BTC/USD"], 21, "long", atr=2.0)
        return b
    on, off = seeded(True), seeded(False)
    assert "BTC/USD" not in on.positions and on.blocked == 1
    assert on.skipped["correlation_filter"] == 1
    assert "BTC/USD" in off.positions and off.blocked == 0


def test_correlation_filter_allows_btc_when_only_one_equity_long():
    b = make_bt(("SPY", "QQQ", "BTC/USD"))
    b.positions["SPY"] = SimPos("SPY", "mean_reversion", "long", 10, 100.0,
                                b.inst["SPY"].df.index[0], 90.0, None, 100.0, None, 1.0)
    b._open(b.inst["BTC/USD"], 21, "long", atr=2.0)
    assert "BTC/USD" in b.positions


def test_short_on_non_shortable_instrument_is_skipped():
    b = make_bt(("USO",))
    b._open(b.inst["USO"], 21, "short", atr=2.0)
    assert not b.positions and b.skipped["short_not_allowed"] == 1


# ------------------------------------------------------------------ metrics
def _trade(pnl):
    ts = pd.Timestamp("2026-03-02", tz="UTC")
    return SimTrade("SPY", "mean_reversion", "long", ts, ts, 100, 100, 1, pnl, "signal", 99)


def _curve(values):
    idx = pd.date_range("2026-03-02 21:00", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_compute_stats_known_values():
    trades = [_trade(300), _trade(-100), _trade(100), _trade(-200)]
    curve = _curve([100_000, 110_000, 99_000, 104_500, 105_000])
    s = compute_stats(trades, curve, 100_000, False)
    assert s["trades"] == 4
    assert s["win_rate"] == pytest.approx(50.0)
    assert s["avg_win"] == pytest.approx(200.0)
    assert s["avg_loss"] == pytest.approx(-150.0)
    assert s["profit_factor"] == pytest.approx(400 / 300)
    assert s["max_dd_pct"] == pytest.approx((99_000 / 110_000 - 1) * 100)
    assert s["max_dd_usd"] == pytest.approx(-11_000)
    assert s["return_pct"] == pytest.approx(5.0)


def test_profit_factor_edge_cases():
    c = _curve([100_000, 100_100])
    assert compute_stats([_trade(50)], c, 100_000, False)["profit_factor"] == math.inf
    assert math.isnan(compute_stats([], c, 100_000, False)["profit_factor"])


def test_sharpe_sign_follows_returns():
    up = _curve(100_000 * np.cumprod(1 + np.array([0.01, 0.002, 0.012, 0.004, 0.009, 0.001])))
    down = _curve(100_000 * np.cumprod(1 - np.array([0.01, 0.002, 0.012, 0.004, 0.009, 0.001])))
    assert sharpe_ratio(up, 100_000, False) > 0
    assert sharpe_ratio(down, 100_000, False) < 0


def test_sharpe_annualisation_matches_hand_calc():
    rets = np.array([0.01, -0.005, 0.007, 0.002, -0.003])
    curve = _curve(100_000 * np.cumprod(1 + rets))
    expected = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(curve, 100_000, False) == pytest.approx(expected, rel=1e-6)


def test_flat_curve_sharpe_is_nan_not_crash():
    assert math.isnan(sharpe_ratio(_curve([100_000] * 5), 100_000, False))

"""Entry point: continuous multi-strategy trading loop.

Run from the project root:
    python -m bot.main        (or)        python bot/main.py
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from alpaca_trade_api.rest import REST, APIError  # noqa: E402
from bot import indicators  # noqa: E402
from bot.api_utils import AUTH_STATUS, status_of  # noqa: E402
from bot.data_feed import MarketData, bar_seconds  # noqa: E402
from bot.portfolio import Portfolio  # noqa: E402
from bot.risk_manager import RiskManager  # noqa: E402
from bot.status import build_snapshot, position_view, write_snapshot  # noqa: E402
from bot.strategies import STRATEGY_CLASSES  # noqa: E402

log = logging.getLogger("bot")

SCHEDULE_OFFSET_S = 5          # run just after a candle closes
FAILURE_RETRY_S = 60
MAX_ENTRY_ATTEMPTS = 3         # per symbol per bar, when an entry order does not fill


@dataclass
class Job:
    name: str
    interval_s: int
    fn: Callable[[], bool]
    aligned: bool = True
    next_run: float = 0.0

    def schedule_next(self, ok: bool) -> None:
        now = time.time()
        if not ok:
            self.next_run = now + FAILURE_RETRY_S
        elif self.aligned:
            self.next_run = (now // self.interval_s + 1) * self.interval_s + SCHEDULE_OFFSET_S
        else:
            self.next_run = now + self.interval_s


class TradingBot:
    def __init__(self, api=None, market_data=None, portfolio=None):
        self.api = api or REST(config.API_KEY, config.API_SECRET, config.BASE_URL,
                               api_version="v2")
        self.md = market_data or MarketData(self.api)
        self.portfolio = portfolio or Portfolio(self.api, config.INSTRUMENTS, self.md)
        self.risk = RiskManager()
        self.strategies = {name: STRATEGY_CLASSES[name](params)
                           for name, params in config.STRATEGY_PARAMS.items()}
        self._stop = False
        self._entry_attempts: Dict[tuple, int] = {}
        self.last_signals: Dict[str, dict] = {}      # latest decision per symbol, for the monitor
        self._last_price: Dict[str, float] = {}
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.status_path = config.STATUS_FILE
        for inst in config.INSTRUMENTS.values():
            strat = self.strategies[inst.strategy]
            self.portfolio.cooldown_seconds[inst.symbol] = (
                config.COOLDOWN_BARS_AFTER_STOP * bar_seconds(strat.timeframe))

    # ------------------------------------------------------------ strategy run
    def run_strategy_cycle(self, strategy_name: str) -> bool:
        """Evaluate every instrument of one strategy. One bad symbol never blocks
        the others. Returns False if any instrument failed (=> quick retry)."""
        strat = self.strategies[strategy_name]
        ok = True
        for inst in config.INSTRUMENTS.values():
            if inst.strategy != strategy_name:
                continue
            try:
                self._process_instrument(inst, strat)
            except Exception as exc:  # noqa: BLE001
                ok = False
                log.error("[%s] %s cycle failed: %s: %s", strategy_name, inst.symbol,
                          type(exc).__name__, exc, exc_info=log.isEnabledFor(logging.DEBUG))
        return ok

    def _process_instrument(self, inst, strat) -> None:
        pf = self.portfolio
        if inst.asset_class == config.EQUITY and not pf.market_open():
            log.debug("[%s] market closed, skipping %s", strat.name, inst.symbol)
            return

        bars = self.md.get_bars(inst, strat.timeframe, strat.lookback_days)
        if bars.empty:
            log.warning("[%s] no bars returned for %s", strat.name, inst.symbol)
            return
        bar_ts = bars.index[-1].isoformat()
        if pf.last_bar(inst.symbol) == bar_ts:
            return                                   # this candle was already acted on

        pos = pf.positions.get(inst.symbol)
        signal_ = strat.evaluate(inst.symbol, bars, pos.direction if pos else None)
        atr_now = float(indicators.atr(bars, config.ATR_PERIOD).iloc[-1])
        log.info("[%s] %s bar=%s -> %s (%s)", strat.name, inst.symbol, bar_ts,
                 "HOLD" if signal_.is_hold else f"exit={signal_.exit} enter={signal_.enter}",
                 signal_.reason)
        self.last_signals[inst.symbol] = {
            "strategy": strat.name, "bar": bar_ts, "reason": signal_.reason,
            "decision": "HOLD" if signal_.is_hold else f"exit={signal_.exit} enter={signal_.enter}",
            "info": signal_.info, "evaluated_at": datetime.now(timezone.utc).isoformat()}

        if pos and pos.trailing_mult:
            last = bars.iloc[-1]
            self.risk.update_trailing_stop(
                pos, float(last["high"] if pos.direction == "long" else last["low"]), atr_now)
            pf.save_state()

        if signal_.exit and inst.symbol in pf.positions:
            pf.close_position(inst.symbol, reason=f"signal: {signal_.reason}")
        if signal_.enter and inst.symbol not in pf.positions:
            if not self._try_enter(inst, strat, signal_.enter, atr_now,
                                   float(bars["close"].iloc[-1]), signal_.reason):
                # The order was sent but did not fill (or was rejected). Leave the bar
                # un-acted so the next poll retries, up to MAX_ENTRY_ATTEMPTS.
                key = (inst.symbol, bar_ts)
                self._entry_attempts[key] = self._entry_attempts.get(key, 0) + 1
                if self._entry_attempts[key] < MAX_ENTRY_ATTEMPTS:
                    log.warning("[%s] %s entry not completed (attempt %d/%d); will retry next poll",
                                strat.name, inst.symbol, self._entry_attempts[key], MAX_ENTRY_ATTEMPTS)
                    return
                log.error("[%s] %s entry failed %d times for bar %s; giving up until the next bar",
                          strat.name, inst.symbol, MAX_ENTRY_ATTEMPTS, bar_ts)
        pf.set_last_bar(inst.symbol, bar_ts)

    def _try_enter(self, inst, strat, direction: str, atr_now: float,
                   bar_close: float, reason: str) -> bool:
        """True when the entry is settled (filled, or deliberately skipped);
        False when an order was attempted but did not fill / was rejected."""
        pf = self.portfolio
        sym = inst.symbol
        if direction == "short" and not inst.allow_short:
            log.info("%s short signal ignored: instrument cannot be sold short on Alpaca", sym)
            return True
        if pf.in_cooldown(sym):
            log.info("%s %s entry skipped: cooling down after a stop-out", sym, direction)
            return True
        allowed, why = self.risk.correlation_allows(sym, direction, pf.positions)
        if not allowed:
            log.info("%s entry blocked - %s", sym, why)
            return True
        if not (atr_now > 0):
            log.warning("%s ATR unavailable; no entry", sym)
            return True

        price = self.md.last_price(inst)
        acct = pf.account()
        bp = acct.non_marginable_buying_power if inst.asset_class == config.CRYPTO \
            else acct.buying_power
        size = self.risk.position_size(inst, price, atr_now, acct.equity, bp,
                                       pf.gross_exposure(), strat.stop_atr_mult)
        if size is None:
            log.warning("%s sizing produced no tradable quantity (price=%.2f atr=%.4f)",
                        sym, price, atr_now)
            return True
        log.info("%s sizing: qty=%s notional=$%.0f risk=$%.2f atr=%.4f%s", sym, size.qty,
                 size.notional, size.risk_dollars, atr_now,
                 f" [capped by {size.capped_by}]" if size.capped_by else "")
        try:
            return pf.open_position(inst, direction, size.qty, size.risk_dollars, atr_now,
                                    strat.name, strat.trailing_atr_mult) is not None
        except APIError as exc:
            # e.g. 403 "asset is not shortable" / insufficient buying power
            log.error("%s %s order rejected: %s", sym, direction, exc)
            return False

    # ---------------------------------------------------------------- risk tick
    def risk_tick(self) -> bool:
        """Fast loop: broker stop fills, software hard stop, trailing stops, daily P&L."""
        pf = self.portfolio
        ok = True
        pf.poll_stop_fills()
        for sym, pos in list(pf.positions.items()):
            inst = config.INSTRUMENTS[sym]
            try:
                if inst.asset_class == config.EQUITY and not pf.market_open():
                    continue
                price = self.md.last_price(inst)
                if self.risk.hard_stop_hit(pos, price):
                    pf.close_position(sym, reason="hard_stop (software)", price_hint=price)
                    continue
                if pos.trailing_mult:
                    before = pos.trailing_stop
                    self.risk.update_trailing_stop(pos, price)
                    if self.risk.trailing_stop_hit(pos, price):
                        pf.close_position(sym, reason="trailing_stop", price_hint=price)
                    elif pos.trailing_stop != before:
                        pf.save_state()
            except Exception as exc:  # noqa: BLE001
                ok = False
                log.error("risk tick failed for %s: %s: %s", sym, type(exc).__name__, exc)
        try:
            acct = pf.account()
            pf.update_daily_pnl(acct.equity)
        except Exception as exc:  # noqa: BLE001
            ok = False
            log.error("daily P&L update failed: %s", exc)
        else:
            self._publish_status(acct)
        return ok

    def _publish_status(self, acct) -> None:
        """Write status.json for the monitor page. Best effort: never affects trading."""
        try:
            pf = self.portfolio
            views = []
            for sym, pos in list(pf.positions.items()):
                try:
                    self._last_price[sym] = self.md.last_price(config.INSTRUMENTS[sym])
                except Exception:  # noqa: BLE001  (keep the last known price)
                    pass
                sma = (self.last_signals.get(sym, {}).get("info") or {}).get("sma")
                views.append(position_view(pos, self._last_price.get(sym), sma))
            try:
                market_open = pf.market_open()
            except Exception:  # noqa: BLE001
                market_open = None
            snap = build_snapshot(
                started_at=self._started_at, mode="PAPER" if config.is_paper() else "LIVE",
                market_open=market_open, equity=acct.equity, cash=acct.cash, positions=views,
                signals=self.last_signals)
            write_snapshot(self.status_path, snap)
        except Exception as exc:  # noqa: BLE001
            log.debug("status snapshot skipped: %s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------ startup
    def _wait_for_connection(self) -> None:
        delay = 5
        while not self._stop:
            try:
                acct = self.portfolio.account()
                log.info("Connected to Alpaca (%s) | equity=$%.2f buying_power=$%.2f",
                         "PAPER" if config.is_paper() else "LIVE", acct.equity, acct.buying_power)
                return
            except APIError as exc:
                if status_of(exc) in AUTH_STATUS:
                    log.critical("Alpaca rejected the credentials (%s). Check .env.", exc)
                    sys.exit(2)
                log.error("Account check failed: %s; retrying in %ss", exc, delay)
            except Exception as exc:  # noqa: BLE001
                log.error("Cannot reach Alpaca (%s: %s); retrying in %ss",
                          type(exc).__name__, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)

    def _startup(self) -> None:
        self._wait_for_connection()
        if self._stop:
            return
        self._warn_on_asset_flags()
        orphans = self.portfolio.reconcile()
        for sym, bp in orphans.items():
            inst = config.INSTRUMENTS[sym]
            strat = self.strategies[inst.strategy]
            try:
                bars = self.md.get_bars(inst, strat.timeframe, strat.lookback_days)
                atr_now = float(indicators.atr(bars, config.ATR_PERIOD).iloc[-1])
                equity = self.portfolio.account().equity
                self.portfolio.adopt_position(inst, bp, atr_now, strat.name,
                                              strat.trailing_atr_mult,
                                              equity * config.RISK_PER_TRADE_PCT)
            except Exception as exc:  # noqa: BLE001
                log.critical("could not adopt untracked %s position: %s -- manage it manually!",
                             sym, exc)
        log.info("Tracking %d open position(s): %s", len(self.portfolio.positions),
                 ", ".join(self.portfolio.positions) or "none")
        self._seed_signals()

    def _seed_signals(self) -> None:
        """Fill `last_signals` once at startup, for display only: after a restart the
        current bar has usually been acted on already, so no decision would be recorded
        (and the monitor would show nothing) until the next bar. Never places orders."""
        for inst in config.INSTRUMENTS.values():
            try:
                strat = self.strategies[inst.strategy]
                bars = self.md.get_bars(inst, strat.timeframe, strat.lookback_days)
                if bars.empty:
                    continue
                pos = self.portfolio.positions.get(inst.symbol)
                sig = strat.evaluate(inst.symbol, bars, pos.direction if pos else None)
                self.last_signals[inst.symbol] = {
                    "strategy": strat.name, "bar": bars.index[-1].isoformat(), "reason": sig.reason,
                    "decision": "HOLD" if sig.is_hold else f"exit={sig.exit} enter={sig.enter}",
                    "info": sig.info, "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "seeded": True}
            except Exception as exc:  # noqa: BLE001  (informational only)
                log.debug("signal seed skipped for %s: %s", inst.symbol, exc)

    def _warn_on_asset_flags(self) -> None:
        for inst in config.INSTRUMENTS.values():
            try:
                asset = self.api.get_asset(inst.broker_symbol)
                if not asset.tradable:
                    log.warning("%s is not tradable on this account", inst.symbol)
                if inst.allow_short and not getattr(asset, "shortable", False):
                    log.warning("%s is not shortable; short signals will be rejected", inst.symbol)
            except Exception as exc:  # noqa: BLE001  (informational only)
                log.debug("asset check skipped for %s: %s", inst.symbol, exc)

    # --------------------------------------------------------------------- loop
    def _build_jobs(self) -> List[Job]:
        jobs = [Job(f"strategy:{name}", strat.check_interval_s,
                    (lambda n=name: self.run_strategy_cycle(n)))
                for name, strat in self.strategies.items()
                if name in config.active_strategies()]
        jobs.append(Job("risk_tick", config.RISK_TICK_SECONDS, self.risk_tick, aligned=False))
        return jobs

    def _run_job(self, job: Job) -> None:
        try:
            ok = job.fn()
        except Exception as exc:  # noqa: BLE001  - the loop must never die
            log.error("job %s crashed: %s: %s", job.name, type(exc).__name__, exc,
                      exc_info=log.isEnabledFor(logging.DEBUG))
            ok = False
        job.schedule_next(ok)

    def stop(self, *_args) -> None:
        log.info("Shutdown requested")
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self._startup()
        jobs = self._build_jobs()
        log.info("Bot running. Open positions keep their broker-side hard stops while the bot is down.")
        while not self._stop:
            now = time.time()
            for job in jobs:
                if self._stop:
                    break
                if now >= job.next_run:
                    self._run_job(job)
            time.sleep(1.0)
        self._shutdown()

    def _shutdown(self) -> None:
        try:
            self.portfolio.update_daily_pnl(self.portfolio.account().equity, force=True)
            self.portfolio.save_state()
        except Exception as exc:  # noqa: BLE001
            log.error("final save failed: %s", exc)
        log.info("Stopped. Positions were left open; their hard stops remain at the broker.")


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL.upper())
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = RotatingFileHandler(config.LOG_FILE, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    error = config.validate()
    if error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(2)
    setup_logging()
    TradingBot().run()


if __name__ == "__main__":
    main()

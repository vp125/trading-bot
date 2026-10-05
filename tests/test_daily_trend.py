"""Daily trend strategy: signals, ATR-multiple stop sizing, shipped config, order flow.
Runs against the real config.INSTRUMENTS (no legacy mapping)."""
import numpy as np
import pandas as pd
import pytest

import config
from bot import indicators as ind
from bot.backtest import Backtester
from bot.data_feed import bar_seconds, to_timeframe
from bot.main import TradingBot
from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager
from bot.strategies import STRATEGY_CLASSES, DailyTrendStrategy
from test_end_to_end import FakeAPI, FakeMD, make_bars, rows

PARAMS = config.STRATEGY_PARAMS["daily_trend"]


def daily(closes, end="2026-10-01"):
    return make_bars(closes, spread=0.25, freq="1D", end=end)


def rising(n=260, start=100.0, step=0.1):
    return start + step * np.arange(n)


def falling(n=260, start=130.0, step=0.1):
    return start - step * np.arange(n)


# ------------------------------------------------------------------ shipped config
def test_shipped_config_is_consistent():
    for inst in config.INSTRUMENTS.values():
        assert inst.strategy in config.STRATEGY_PARAMS and inst.strategy in STRATEGY_CLASSES
    for sym in ("SPY", "QQQ", "GLD"):
        assert config.INSTRUMENTS[sym].strategy == "daily_trend"
        assert config.INSTRUMENTS[sym].allow_short is False
    assert "daily_trend" in config.active_strategies()
    assert "mean_reversion" not in config.active_strategies()
    assert bar_seconds(tuple(PARAMS["timeframe"])) == 86400
    assert to_timeframe(tuple(PARAMS["timeframe"])).unit.value == "Day"
    # enough calendar days of history to get min_bars trading days
    assert PARAMS["lookback_days"] * 5 / 7 >= PARAMS["min_bars"]


# ------------------------------------------------------------------ strategy signals
def test_enters_long_when_flat_and_close_above_sma():
    s = DailyTrendStrategy(PARAMS)
    sig = s.evaluate("SPY", daily(rising()), None)
    assert sig.enter == "long" and not sig.exit


def test_stays_flat_below_sma_and_never_shorts():
    s = DailyTrendStrategy(PARAMS)
    sig = s.evaluate("SPY", daily(falling()), None)
    assert sig.is_hold and sig.enter is None


def test_holds_then_exits_when_close_drops_below_sma():
    s = DailyTrendStrategy(PARAMS)
    assert s.evaluate("SPY", daily(rising()), "long").is_hold
    closes = np.append(rising(), 90.0)                    # crash through the 200-day SMA
    sig = s.evaluate("SPY", daily(closes), "long")
    assert sig.exit is True and sig.enter is None


def test_waits_for_enough_bars():
    s = DailyTrendStrategy(PARAMS)
    sig = s.evaluate("SPY", daily(rising(PARAMS["min_bars"] - 1)), None)
    assert sig.is_hold and "need" in sig.reason


# ------------------------------------------------------------------ sizing
def test_stop_atr_mult_shrinks_qty_but_keeps_loss_at_one_percent():
    rm, eq = RiskManager(), 100_000.0
    spy = config.INSTRUMENTS["SPY"]
    one = rm.position_size(spy, 100.0, 2.0, eq, 400_000, 0.0)
    three = rm.position_size(spy, 100.0, 2.0, eq, 400_000, 0.0, stop_atr_mult=3.0)
    assert one.qty == 500 and three.qty == 166                # 1000 / 2 vs 1000 / 6, whole shares
    assert three.stop_distance == pytest.approx(1000 / 166)
    assert three.qty * three.stop_distance <= 1000.0 + 1e-9    # never more than 1% of equity
    assert rm.position_size(spy, 100.0, 2.0, eq, 400_000, 0.0, stop_atr_mult=0) is None


# ------------------------------------------------------------------ backtest engine
def test_backtest_sizes_daily_trend_with_three_atr_stop():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    c = np.full(n, 100.0)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)
    b = Backtester({"SPY": df}, idx[250], ["SPY"], 100_000.0, 0.0005)
    b._open(b.inst["SPY"], 251, "long", atr=2.0)
    pos = b.positions["SPY"]
    assert pos.qty == 166
    assert (pos.entry_price - pos.hard_stop_price) * pos.qty == pytest.approx(1000.0, rel=0.01)
    assert pos.hard_stop_price == pytest.approx(pos.entry_price - 1000 / 166, abs=0.01)


# ------------------------------------------------------------------ order flow
@pytest.fixture
def env(tmp_path):
    api = FakeAPI()
    for s, p in {"SPY": 126.0, "QQQ": 100.0, "GLD": 100.0, "USO": 100.0, "BTCUSD": 100_000.0}.items():
        api.prices[s] = p
    md = FakeMD(api)
    pf = Portfolio(api, config.INSTRUMENTS, md, state_path=tmp_path / "state.json",
                   trades_csv=tmp_path / "trades.csv", pnl_csv=tmp_path / "pnl.csv")
    bot = TradingBot(api=api, market_data=md, portfolio=pf)
    md.bars["SPY"] = daily(rising())
    md.bars["QQQ"] = daily(falling())
    md.bars["GLD"] = daily(falling())
    return type("Env", (), dict(api=api, md=md, pf=pf, bot=bot, tmp=tmp_path))


def test_entry_places_three_atr_broker_stop_then_signal_exit_logs_trade(env):
    assert env.bot.run_strategy_cycle("daily_trend") is True
    assert set(env.pf.positions) == {"SPY"}                    # QQQ / GLD below their SMA: flat
    pos = env.pf.positions["SPY"]
    atr = float(ind.atr(env.md.bars["SPY"], config.ATR_PERIOD).iloc[-1])
    assert pos.direction == "long" and pos.trailing_mult is None
    assert pos.qty == int(1000 / (3 * atr)) or pos.qty == pytest.approx(1000 / (3 * atr), abs=1)
    assert (pos.entry_price - pos.hard_stop_price) * pos.qty <= 1000.0 + 1e-6
    stops = [o for o in env.api.submitted if o["type"] == "stop"]
    assert len(stops) == 1 and float(stops[0]["stop_price"]) == pytest.approx(pos.hard_stop_price, abs=0.01)

    env.bot.run_strategy_cycle("daily_trend")                  # same bar again: no second order
    assert len([o for o in env.api.submitted if o["side"] == "buy" and o["type"] == "market"]) == 1

    crash = np.append(rising(), 90.0)                          # next daily bar closes below the SMA
    env.md.bars["SPY"] = daily(crash, end="2026-10-02")
    env.api.set_price("SPY", 90.0, trigger_stops=False)
    assert env.bot.run_strategy_cycle("daily_trend") is True
    assert "SPY" not in env.pf.positions
    t = rows(env.tmp / "trades.csv")
    assert len(t) == 1 and t[0]["instrument"] == "SPY" and t[0]["direction"] == "long"
    assert float(t[0]["profit_loss"]) < 0


def test_only_active_strategies_get_scheduled_jobs(env):
    names = {j.name for j in env.bot._build_jobs()}
    assert "strategy:daily_trend" in names and "strategy:mean_reversion" not in names
    assert "risk_tick" in names


def test_btc_is_disabled_in_shipped_config():
    assert "BTC/USD" not in config.INSTRUMENTS
    assert "momentum_breakout" not in config.active_strategies()
    assert set(config.INSTRUMENTS) == {"SPY", "QQQ", "GLD", "USO"}

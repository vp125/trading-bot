"""Offline tests: indicators, the three strategies, sizing, stops, correlation filter."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from bot import indicators as ind
from bot.risk_manager import RiskManager, hard_stop_price
from bot.strategies import (MeanReversionStrategy, MomentumBreakoutStrategy,
                            TrendFollowingStrategy)

INSTR = config.INSTRUMENTS
pytestmark = pytest.mark.usefixtures("legacy_instruments")


def make_bars(closes, volumes=None, spread=0.25, freq="15min"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(end="2026-10-01 20:00", periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "open": closes, "high": closes + spread, "low": closes - spread, "close": closes,
        "volume": np.full(len(closes), 100.0) if volumes is None else np.asarray(volumes, float),
    }, index=idx)


# ------------------------------------------------------------------ indicators
def test_atr_constant_range_is_that_range():
    df = make_bars(np.full(60, 100.0), spread=1.0)          # high-low = 2, no gaps
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_ema_converges_to_constant():
    s = pd.Series(np.full(300, 50.0))
    assert ind.ema(s, 50).iloc[-1] == pytest.approx(50.0)


# -------------------------------------------------------------- mean reversion
def _mr(symbol):
    return MeanReversionStrategy(config.STRATEGY_PARAMS["mean_reversion"]), symbol


def _closes_with_last(last):
    base = [100 + (0.5 if i % 2 else -0.5) for i in range(39)]
    return base + [last]


def _z_of(last):
    c = pd.Series(_closes_with_last(last))
    return ((c - ind.sma(c, 20)) / ind.rolling_std(c, 20)).iloc[-1]


def test_mean_reversion_thresholds_differ_between_spy_and_qqq():
    # find a last price whose z is between -1.8 and -1.5 (SPY trades, QQQ does not)
    last = next(x for x in np.arange(100, 90, -0.01) if -1.75 < _z_of(x) < -1.55)
    bars = make_bars(_closes_with_last(last))
    strat, _ = _mr("SPY")
    assert strat.evaluate("SPY", bars, None).enter == "long"
    assert strat.evaluate("QQQ", bars, None).enter is None


def test_mean_reversion_short_and_exit():
    strat, _ = _mr("SPY")
    hi = make_bars(_closes_with_last(110))
    assert strat.evaluate("SPY", hi, None).enter == "short"
    # exits are level based
    at_mean = make_bars(_closes_with_last(100.2))
    assert strat.evaluate("SPY", at_mean, "long").exit is True       # >= SMA
    assert strat.evaluate("SPY", at_mean, "short").exit is False     # price still above the SMA


def test_mean_reversion_short_exit_when_price_below_mean():
    strat, _ = _mr("SPY")
    bars = make_bars(_closes_with_last(99.5))
    assert strat.evaluate("SPY", bars, "short").exit is True
    assert strat.evaluate("SPY", bars, "long").exit is False


# ------------------------------------------------------------------- momentum
def _mo_bars(last_close, last_vol):
    closes = [100 + 0.3 * math.sin(i) for i in range(30)] + [last_close]
    vols = [100.0] * 30 + [last_vol]
    return make_bars(closes, vols, spread=0.2, freq="1h")


def test_momentum_requires_volume_confirmation():
    s = MomentumBreakoutStrategy(config.STRATEGY_PARAMS["momentum_breakout"])
    assert s.evaluate("BTC/USD", _mo_bars(105, 200), None).enter == "long"       # 2.0x vol
    assert s.evaluate("BTC/USD", _mo_bars(105, 149), None).is_hold               # 1.49x vol
    assert s.evaluate("BTC/USD", _mo_bars(105, 150), None).enter == "long"       # exactly 1.5x


def test_momentum_breakdown_exits_long():
    s = MomentumBreakoutStrategy(config.STRATEGY_PARAMS["momentum_breakout"])
    sig = s.evaluate("BTC/USD", _mo_bars(95, 200), "long")
    assert sig.exit is True
    assert s.trailing_atr_mult == 2.0


# ---------------------------------------------------------------------- trend
def _trend_bars():
    t = np.arange(900)
    closes = 100 + 25 * np.sin(2 * np.pi * t / 330) + 0.01 * t
    return make_bars(closes, freq="4h")


def test_trend_enters_only_on_cross_and_exits_level_based():
    bars = _trend_bars()
    s = TrendFollowingStrategy(config.STRATEGY_PARAMS["trend_following"])
    fast, slow = ind.ema(bars["close"], 50), ind.ema(bars["close"], 200)
    up = [i for i in range(300, len(bars)) if fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]]
    down = [i for i in range(300, len(bars)) if fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]]
    assert up and down
    i_up, i_down = up[0], down[0]

    assert s.evaluate("GLD", bars.iloc[: i_up + 1], None).enter == "long"       # on the cross bar
    assert s.evaluate("GLD", bars.iloc[: i_up], None).is_hold                    # bar before
    assert s.evaluate("GLD", bars.iloc[: i_up + 3], None).is_hold                # after the cross: no late entry

    assert s.evaluate("GLD", bars.iloc[: i_down + 1], None).enter == "short"
    # a long held through a cross-down is exited (and may flip short on the cross bar)
    sig = s.evaluate("GLD", bars.iloc[: i_down + 1], "long")
    assert sig.exit and sig.enter == "short"
    # level-based: still exits several bars later (e.g. a missed bar)
    assert s.evaluate("GLD", bars.iloc[: i_down + 4], "long").exit is True
    assert s.trailing_atr_mult == 3.0


def test_trend_needs_warmup():
    bars = _trend_bars().iloc[:100]
    s = TrendFollowingStrategy(config.STRATEGY_PARAMS["trend_following"])
    assert s.evaluate("GLD", bars, None).is_hold


# ----------------------------------------------------------------------- risk
rm = RiskManager()
EQ = 100_000.0


def test_atr_sizing_gives_constant_risk_per_atr():
    spy, gld = INSTR["SPY"], INSTR["GLD"]
    quiet = rm.position_size(spy, price=100, atr=10.0, equity=EQ, buying_power=400_000, current_gross=0)
    wild = rm.position_size(gld, price=100, atr=20.0, equity=EQ, buying_power=400_000, current_gross=0)
    assert quiet.capped_by is None and wild.capped_by is None
    assert quiet.qty * 10.0 == pytest.approx(1000)       # 1 ATR == 1% of equity
    assert wild.qty * 20.0 == pytest.approx(1000)
    assert quiet.qty > wild.qty                           # quieter instrument -> larger position


def test_cap_reduces_size_but_stop_still_risks_exactly_one_percent():
    r = rm.position_size(INSTR["SPY"], price=100, atr=0.5, equity=EQ, buying_power=400_000, current_gross=0)
    assert r.capped_by == "position_cap"
    assert r.notional <= EQ * config.MAX_POSITION_NOTIONAL_PCT
    assert r.qty * r.stop_distance == pytest.approx(1000)   # loss at stop == 1% of equity


def test_gross_and_buying_power_caps():
    g = rm.position_size(INSTR["SPY"], 100, 0.5, EQ, 400_000, current_gross=180_000)
    assert g.capped_by == "gross_cap" and g.notional <= 20_000
    b = rm.position_size(INSTR["SPY"], 100, 0.5, EQ, buying_power=10_000, current_gross=0)
    assert b.capped_by == "buying_power" and b.notional <= 10_000


def test_whole_shares_for_equities_fractional_for_crypto_and_loss_never_exceeds_1pct():
    r = rm.position_size(INSTR["GLD"], 101.3, 3.0, EQ, 400_000, 0)
    assert r.qty == int(r.qty)
    stop = hard_stop_price("long", 101.3, r.qty, r.risk_dollars)
    assert (101.3 - stop) * r.qty <= 1000 + 1e-6
    c = rm.position_size(INSTR["BTC/USD"], 100_000, 1_500, EQ, 100_000, 0)
    assert c.qty == pytest.approx(round(c.qty, 6)) and c.qty != int(c.qty)
    assert c.qty * c.stop_distance == pytest.approx(1000, rel=1e-3)


def test_hard_stop_direction():
    assert hard_stop_price("long", 100, 200, 1000) == 95.0
    assert hard_stop_price("short", 100, 200, 1000) == 105.0


def test_sizing_rejects_bad_inputs():
    assert rm.position_size(INSTR["SPY"], 100, float("nan"), EQ, 1e5, 0) is None
    assert rm.position_size(INSTR["SPY"], 100, 0.0, EQ, 1e5, 0) is None
    assert rm.position_size(INSTR["SPY"], 100, 1.0, EQ, buying_power=50, current_gross=0) is None


def test_correlation_filter():
    pos = {"SPY": SimpleNamespace(direction="long"), "QQQ": SimpleNamespace(direction="long")}
    assert rm.correlation_allows("BTC/USD", "long", pos)[0] is False
    assert rm.correlation_allows("BTC/USD", "short", pos)[0] is True
    assert rm.correlation_allows("GLD", "long", pos)[0] is True
    assert rm.correlation_allows("BTC/USD", "long", {"SPY": pos["SPY"]})[0] is True
    pos["QQQ"] = SimpleNamespace(direction="short")
    assert rm.correlation_allows("BTC/USD", "long", pos)[0] is True


def _pos(direction, entry=100.0, atr=2.0, mult=3.0):
    return SimpleNamespace(direction=direction, trailing_mult=mult, extreme_price=entry,
                           trailing_stop=entry - mult * atr if direction == "long" else entry + mult * atr,
                           last_atr=atr, hard_stop_price=entry - 5 if direction == "long" else entry + 5)


def test_trailing_stop_ratchets_up_only_for_long():
    p = _pos("long")
    assert p.trailing_stop == 94
    rm.update_trailing_stop(p, 110)
    assert p.trailing_stop == 104
    rm.update_trailing_stop(p, 105)                       # pullback: stop must not drop
    assert p.trailing_stop == 104
    assert not rm.trailing_stop_hit(p, 104.5)
    assert rm.trailing_stop_hit(p, 103.9)


def test_trailing_stop_does_not_loosen_when_atr_expands():
    p = _pos("long")
    rm.update_trailing_stop(p, 110)                       # stop 104 with ATR 2
    rm.update_trailing_stop(p, 108, atr=4.0)              # wider ATR would imply 98
    assert p.trailing_stop == 104
    q = _pos("short")
    rm.update_trailing_stop(q, 90)                        # stop 96 with ATR 2
    rm.update_trailing_stop(q, 92, atr=4.0)               # wider ATR would imply 102
    assert q.trailing_stop == 96


def test_trailing_stop_ratchets_down_only_for_short():
    p = _pos("short")
    rm.update_trailing_stop(p, 90)
    assert p.trailing_stop == 96
    rm.update_trailing_stop(p, 95)
    assert p.trailing_stop == 96
    assert rm.trailing_stop_hit(p, 96.1)


def test_hard_stop_hit():
    assert rm.hard_stop_hit(_pos("long"), 95.0) and not rm.hard_stop_hit(_pos("long"), 95.5)
    assert rm.hard_stop_hit(_pos("short"), 105.0) and not rm.hard_stop_hit(_pos("short"), 104.5)


# ------------------------------------------------------------------ data feed
def test_regular_hours_filter_drops_thin_extended_hours_candles():
    from bot.data_feed import regular_hours_only
    # 2026-10-02 (Fri, EDT = UTC-4): 12:15Z = 08:15 ET (pre), 13:30Z = 09:30 ET, 19:45Z = 15:45 ET, 20:15Z = 16:15 ET (post)
    idx = pd.DatetimeIndex(["2026-10-02 12:15", "2026-10-02 13:30", "2026-10-02 19:45", "2026-10-02 20:15"], tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    kept = regular_hours_only(df)
    assert [t.strftime("%H:%M") for t in kept.index] == ["13:30", "19:45"]


def test_uso_not_shortable_and_btc_not_shortable_in_config():
    assert INSTR["USO"].allow_short is False and INSTR["BTC/USD"].allow_short is False
    assert INSTR["GLD"].allow_short is True and INSTR["SPY"].allow_short is True

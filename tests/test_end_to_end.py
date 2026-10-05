"""End-to-end order-flow tests against a fake Alpaca client (no network, no real orders)."""
import csv
import itertools
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from alpaca_trade_api.rest import APIError

import config
from bot.main import TradingBot
from bot.portfolio import Portfolio

INSTR = config.INSTRUMENTS
pytestmark = pytest.mark.usefixtures("legacy_instruments")


def http_error(code):
    return SimpleNamespace(response=SimpleNamespace(status_code=code), request=None)


class FakeAPI:
    """Just enough of alpaca_trade_api.REST to exercise the bot."""

    def __init__(self, equity=100_000.0):
        self.cash = equity
        self.prices = {}
        self.pos = {}                  # broker symbol -> dict(qty, avg)
        self.orders = {}
        self.ids = itertools.count(1)
        self.market_is_open = True
        self.fail_stop_orders = False
        self.submitted = []

    # --- helpers
    def set_price(self, symbol, price, trigger_stops=True):
        self.prices[symbol] = price
        if not trigger_stops:
            return
        for o in list(self.orders.values()):
            if o.status != "new" or o.type not in ("stop", "stop_limit"):
                continue
            bsym = o.symbol.replace("/", "")
            hit = (o.side == "sell" and price <= float(o.stop_price)) or \
                  (o.side == "buy" and price >= float(o.stop_price))
            if hit and bsym in self.pos:
                self._fill(o, float(o.stop_price))
                self._apply(bsym, o.side, float(o.qty), float(o.stop_price))

    def _apply(self, bsym, side, qty, price):
        signed = qty if side == "buy" else -qty
        cur = self.pos.get(bsym, {"qty": 0.0, "avg": price})
        new_qty = cur["qty"] + signed
        if abs(new_qty) < 1e-12:
            self.pos.pop(bsym, None)
        else:
            avg = price if cur["qty"] == 0 else cur["avg"]
            self.pos[bsym] = {"qty": new_qty, "avg": avg}

    def _fill(self, o, price):
        o.status, o.filled_qty, o.filled_avg_price = "filled", str(o.qty), str(price)
        o.filled_at = datetime.now(timezone.utc)

    def _order(self, **kw):
        o = SimpleNamespace(id=f"o{next(self.ids)}", status="new", filled_qty="0",
                            filled_avg_price=None, filled_at=None, **kw)
        self.orders[o.id] = o
        return o

    # --- REST surface
    def get_account(self):
        eq = self.cash + sum(p["qty"] * self.prices[s] - p["qty"] * p["avg"] for s, p in self.pos.items())
        return SimpleNamespace(equity=str(eq), buying_power=str(eq * 4),
                               non_marginable_buying_power=str(self.cash), cash=str(self.cash))

    def get_clock(self):
        return SimpleNamespace(is_open=self.market_is_open)

    def list_positions(self):
        return [self._posobj(s, p) for s, p in self.pos.items()]

    def _posobj(self, s, p):
        px = self.prices[s]
        return SimpleNamespace(symbol=s, qty=str(p["qty"]), avg_entry_price=str(p["avg"]),
                               market_value=str(p["qty"] * px))

    def get_position(self, symbol):
        if symbol not in self.pos:
            raise APIError({"message": "position does not exist"}, http_error(404))
        return self._posobj(symbol, self.pos[symbol])

    def submit_order(self, symbol, qty=None, side="buy", type="market", time_in_force="day",
                     stop_price=None, limit_price=None, **kw):
        self.submitted.append(dict(symbol=symbol, qty=qty, side=side, type=type,
                                   stop_price=stop_price, limit_price=limit_price, tif=time_in_force))
        bsym = symbol.replace("/", "")
        if type in ("stop", "stop_limit"):
            if self.fail_stop_orders:
                raise APIError({"message": "stop rejected"}, http_error(422))
            return self._order(symbol=symbol, qty=qty, side=side, type=type, stop_price=stop_price)
        if side == "sell" and bsym not in self.pos and not INSTR[symbol].allow_short:
            raise APIError({"message": "cannot short"}, http_error(403))
        o = self._order(symbol=symbol, qty=qty, side=side, type=type, stop_price=None)
        px = self.prices[bsym]
        self._fill(o, px)
        self._apply(bsym, side, float(qty), px)
        return o

    def get_order(self, order_id):
        return self.orders[order_id]

    def cancel_order(self, order_id):
        o = self.orders[order_id]
        if o.status != "new":
            raise APIError({"message": "not cancelable"}, http_error(422))
        o.status = "canceled"

    def close_position(self, symbol):
        if symbol not in self.pos:
            raise APIError({"message": "position does not exist"}, http_error(404))
        held = [o for o in self.orders.values() if o.status == "new" and o.symbol.replace("/", "") == symbol]
        if held:
            raise APIError({"message": "insufficient qty available (held for orders)"}, http_error(403))
        p = self.pos[symbol]
        side = "sell" if p["qty"] > 0 else "buy"
        sym = next(i.symbol for i in INSTR.values() if i.broker_symbol == symbol)
        return self.submit_order(sym, abs(p["qty"]), side, "market")

    def get_asset(self, symbol):
        return SimpleNamespace(tradable=True, shortable=True)


class FakeMD:
    def __init__(self, api):
        self.api, self.bars = api, {}

    def get_bars(self, inst, spec, lookback_days, now=None):
        return self.bars[inst.symbol]

    def last_price(self, inst):
        return self.api.prices[inst.broker_symbol]


def make_bars(closes, volumes=None, spread=0.25, freq="15min", end="2026-10-01 20:00"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(end=end, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes + spread, "low": closes - spread,
                         "close": closes,
                         "volume": np.full(len(closes), 100.0) if volumes is None else np.asarray(volumes, float)},
                        index=idx)


def oversold_spy(last=96.0):
    return make_bars([100 + (0.5 if i % 2 else -0.5) for i in range(39)] + [last])


def btc_breakout(direction="up", end="2026-10-01 20:00"):
    closes = [100_000 + 300 * math.sin(i) for i in range(30)] + [102_500 if direction == "up" else 97_500]
    vols = [10.0] * 30 + [30.0]
    return make_bars(closes, vols, spread=150, freq="1h", end=end)


@pytest.fixture
def env(tmp_path):
    api = FakeAPI()
    for s, p in {"SPY": 100.0, "QQQ": 100.0, "GLD": 100.0, "USO": 100.0, "BTCUSD": 100_000.0}.items():
        api.prices[s] = p
    md = FakeMD(api)
    pf = Portfolio(api, INSTR, md, state_path=tmp_path / "state.json",
                   trades_csv=tmp_path / "trades.csv", pnl_csv=tmp_path / "pnl.csv")
    bot = TradingBot(api=api, market_data=md, portfolio=pf)
    return SimpleNamespace(api=api, md=md, pf=pf, bot=bot, tmp=tmp_path)


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_cycle(env, name):
    assert env.bot.run_strategy_cycle(name) is True


def test_mean_reversion_entry_stop_and_exit_logs_trade(env):
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    run_cycle(env, "mean_reversion")

    pos = env.pf.positions["SPY"]
    assert pos.direction == "long" and pos.qty == int(pos.qty) and pos.qty > 0
    # hard stop sits below entry and risks <= 1% of equity (100k -> $1000), and is a live broker order
    assert (pos.entry_price - pos.hard_stop_price) * pos.qty <= 1000 + 1e-6
    assert (pos.entry_price - pos.hard_stop_price) * pos.qty > 990
    stops = [o for o in env.api.orders.values() if o.type == "stop" and o.status == "new"]
    assert len(stops) == 1 and float(stops[0].stop_price) == pos.hard_stop_price
    assert not (env.tmp / "trades.csv").exists()

    run_cycle(env, "mean_reversion")                       # same candle: nothing happens twice
    assert len([s for s in env.api.submitted if s["type"] == "market"]) == 1

    # price reverts to the mean on the next candle -> exit, stop cancelled, trade logged
    env.api.set_price("SPY", 100.5)
    env.md.bars["SPY"] = make_bars([100 + (0.5 if i % 2 else -0.5) for i in range(40)] + [100.4],
                                   end="2026-10-01 20:15")
    run_cycle(env, "mean_reversion")
    assert "SPY" not in env.pf.positions and "SPY" not in env.api.pos
    assert all(o.status != "new" for o in env.api.orders.values())
    t = rows(env.tmp / "trades.csv")
    assert len(t) == 1 and t[0]["instrument"] == "SPY" and t[0]["direction"] == "long"
    assert float(t[0]["profit_loss"]) == pytest.approx((100.5 - 100.0) * pos.qty, abs=0.01)
    assert list(t[0].keys()) == ["timestamp", "instrument", "direction", "entry_price",
                                 "exit_price", "profit_loss", "position_size"]


def test_broker_hard_stop_fill_is_logged_and_triggers_cooldown(env):
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    run_cycle(env, "mean_reversion")
    pos = env.pf.positions["SPY"]

    env.api.set_price("SPY", pos.hard_stop_price - 0.5)     # broker fires the stop order
    assert env.bot.risk_tick() is True
    assert "SPY" not in env.pf.positions
    t = rows(env.tmp / "trades.csv")
    assert float(t[0]["exit_price"]) == pos.hard_stop_price
    assert float(t[0]["profit_loss"]) == pytest.approx(-1000, abs=pos.qty * 0.01 + 0.01)
    assert env.pf.in_cooldown("SPY")

    # next candle is still oversold, but we are in cooldown -> no re-entry
    env.md.bars["SPY"] = oversold_spy(95.0).set_axis(
        pd.date_range(end="2026-10-01 20:15", periods=40, freq="15min", tz="UTC"))
    run_cycle(env, "mean_reversion")
    assert "SPY" not in env.pf.positions


def test_software_hard_stop_backs_up_a_stop_that_did_not_fire(env):
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    run_cycle(env, "mean_reversion")
    pos = env.pf.positions["SPY"]
    env.api.set_price("SPY", pos.hard_stop_price - 1, trigger_stops=False)   # broker stop "missed"
    env.bot.risk_tick()
    assert "SPY" not in env.pf.positions
    assert rows(env.tmp / "trades.csv")[0]["instrument"] == "SPY"
    assert all(o.status != "new" for o in env.api.orders.values())


def test_entry_is_flattened_if_hard_stop_cannot_be_placed(env):
    env.api.fail_stop_orders = True
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    env.bot.run_strategy_cycle("mean_reversion")
    assert "SPY" not in env.pf.positions and "SPY" not in env.api.pos        # "no exceptions"


def test_btc_long_entry_uses_stop_limit_and_trailing_stop_exit(env):
    env.md.bars["BTC/USD"] = btc_breakout("up")
    run_cycle(env, "momentum_breakout")
    pos = env.pf.positions["BTC/USD"]
    assert pos.direction == "long" and pos.trailing_mult == 2.0 and pos.qty != int(pos.qty)
    order = next(s for s in env.api.submitted if s["type"] == "stop_limit")
    assert order["tif"] == "gtc" and float(order["limit_price"]) < float(order["stop_price"])

    # rally ratchets the trailing stop up; pullback below it exits (before the hard stop)
    env.api.set_price("BTCUSD", 106_000)
    env.bot.risk_tick()
    high_trail = pos.trailing_stop
    assert high_trail > pos.hard_stop_price
    env.api.set_price("BTCUSD", 105_000)
    env.bot.risk_tick()
    assert pos.trailing_stop == high_trail                   # never loosens
    env.api.set_price("BTCUSD", high_trail - 10, trigger_stops=False)
    env.bot.risk_tick()
    assert "BTC/USD" not in env.pf.positions
    assert float(rows(env.tmp / "trades.csv")[0]["profit_loss"]) > 0


def test_correlation_filter_blocks_btc_long_when_spy_and_qqq_long(env):
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = oversold_spy(94.0)
    run_cycle(env, "mean_reversion")
    assert env.pf.direction("SPY") == "long" and env.pf.direction("QQQ") == "long"

    env.md.bars["BTC/USD"] = btc_breakout("up")
    run_cycle(env, "momentum_breakout")
    assert "BTC/USD" not in env.pf.positions
    assert not any(s["symbol"] == "BTC/USD" for s in env.api.submitted)


def test_btc_breakdown_never_opens_a_short_but_exits_a_long(env):
    env.md.bars["BTC/USD"] = btc_breakout("down")
    run_cycle(env, "momentum_breakout")
    assert "BTC/USD" not in env.pf.positions and not env.api.submitted

    env.md.bars["BTC/USD"] = btc_breakout("up")
    env.pf.state["last_bar"].clear()
    run_cycle(env, "momentum_breakout")
    assert env.pf.direction("BTC/USD") == "long"
    env.md.bars["BTC/USD"] = btc_breakout("down", end="2026-10-01 21:00")
    run_cycle(env, "momentum_breakout")
    assert "BTC/USD" not in env.pf.positions and "BTCUSD" not in env.api.pos
    assert rows(env.tmp / "trades.csv")[0]["direction"] == "long"      # exit only; no short was opened
    assert not any(p["qty"] < 0 for p in env.api.pos.values())


def test_equities_skipped_when_market_closed_but_crypto_still_runs(env):
    env.api.market_is_open = False
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    run_cycle(env, "mean_reversion")
    assert not env.api.submitted
    env.md.bars["BTC/USD"] = btc_breakout("up")
    run_cycle(env, "momentum_breakout")
    assert env.pf.direction("BTC/USD") == "long"


def test_state_survives_restart_and_reconciles_with_broker(env):
    env.md.bars["SPY"] = oversold_spy()
    env.md.bars["QQQ"] = make_bars(np.full(40, 100.0))
    run_cycle(env, "mean_reversion")
    qty = env.pf.positions["SPY"].qty

    pf2 = Portfolio(env.api, INSTR, env.md, state_path=env.tmp / "state.json",
                    trades_csv=env.tmp / "trades.csv", pnl_csv=env.tmp / "pnl.csv")
    assert pf2.positions["SPY"].qty == qty
    assert pf2.reconcile() == {}                             # broker agrees

    # position vanishes while the bot is down (stop filled) -> recorded on startup
    env.api.set_price("SPY", pf2.positions["SPY"].hard_stop_price - 1)
    pf3 = Portfolio(env.api, INSTR, env.md, state_path=env.tmp / "state.json",
                    trades_csv=env.tmp / "trades.csv", pnl_csv=env.tmp / "pnl.csv")
    pf3.reconcile()
    assert "SPY" not in pf3.positions
    assert len(rows(env.tmp / "trades.csv")) == 1


def test_untracked_broker_position_is_adopted_with_a_hard_stop(env):
    env.api.pos["GLD"] = {"qty": 50.0, "avg": 100.0}
    orphans = env.pf.reconcile()
    assert list(orphans) == ["GLD"]
    env.pf.adopt_position(INSTR["GLD"], orphans["GLD"], atr=2.0, strategy="trend_following",
                          trailing_mult=3.0, risk_dollars=1000.0)
    pos = env.pf.positions["GLD"]
    assert pos.stop_order_id and pos.hard_stop_price == pytest.approx(80.0)


def test_daily_pnl_csv_rolls_over_by_new_york_date(env):
    ny = timezone(timedelta(hours=-4))
    d1 = datetime(2026, 10, 1, 10, 0, tzinfo=ny)
    d2 = datetime(2026, 10, 2, 10, 0, tzinfo=ny)
    env.pf.update_daily_pnl(100_000, now=d1)
    env.pf.update_daily_pnl(100_450, now=d1 + timedelta(hours=3), force=True)
    env.pf.update_daily_pnl(100_450, now=d2)                 # rollover closes out day 1
    env.pf.update_daily_pnl(99_900, now=d2 + timedelta(hours=1), force=True)
    r = rows(env.tmp / "pnl.csv")
    assert [x["date"] for x in r] == ["2026-10-01", "2026-10-02"]
    assert float(r[0]["pnl"]) == 450.0 and float(r[1]["pnl"]) == -550.0

"""Portfolio: broker interaction, position state, trade log, daily P&L.

Everything that talks to Alpaca about orders/positions/accounts lives here.
Open-position state is persisted to state.json so a restart (or crash) never
loses track of entry prices, stops or trailing levels, and is reconciled
against the broker on startup.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from alpaca_trade_api.rest import APIError

import config
from .api_utils import retry_api, status_of
from .risk_manager import hard_stop_price

log = logging.getLogger(__name__)

TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}

TRADE_FIELDS = ["timestamp", "instrument", "direction", "entry_price",
                "exit_price", "profit_loss", "position_size"]
PNL_FIELDS = ["date", "starting_equity", "ending_equity", "pnl", "pnl_pct",
              "realized_pnl", "trades"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _px(x: float) -> str:
    return f"{round(float(x), 2):.2f}"


@dataclass
class AccountInfo:
    equity: float
    buying_power: float
    non_marginable_buying_power: float
    cash: float


@dataclass
class TrackedPosition:
    symbol: str
    direction: str                      # "long" | "short"
    qty: float
    entry_price: float
    entry_time: str
    strategy: str
    atr_at_entry: float
    hard_stop_price: float
    stop_order_id: Optional[str] = None
    trailing_mult: Optional[float] = None
    extreme_price: float = 0.0          # best price since entry (high for long, low for short)
    trailing_stop: Optional[float] = None
    last_atr: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "TrackedPosition":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class Portfolio:
    def __init__(self, api, instruments, market_data, state_path=None,
                 trades_csv=None, pnl_csv=None):
        self.api = api
        self.instruments = instruments
        self.md = market_data
        self.state_path = state_path or config.STATE_FILE
        self.trades_csv = trades_csv or config.TRADES_CSV
        self.pnl_csv = pnl_csv or config.DAILY_PNL_CSV
        self.cooldown_seconds: Dict[str, int] = {}
        self.positions: Dict[str, TrackedPosition] = {}
        self.state: dict = {"positions": {}, "cooldowns": {}, "last_bar": {}, "day": {}}
        self._clock_cache = (0.0, True)
        self._last_pnl_write = 0.0
        self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> None:
        try:
            with open(self.state_path) as fh:
                loaded = json.load(fh)
            self.state.update(loaded)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s (%s); starting with empty state", self.state_path, exc)
            return
        for sym, d in self.state.get("positions", {}).items():
            self.positions[sym] = TrackedPosition.from_dict(d)

    def save_state(self) -> None:
        self.state["positions"] = {s: asdict(p) for s, p in self.positions.items()}
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=2, default=float)
        os.replace(tmp, self.state_path)

    def direction(self, symbol: str) -> Optional[str]:
        pos = self.positions.get(symbol)
        return pos.direction if pos else None

    def last_bar(self, symbol: str) -> Optional[str]:
        return self.state["last_bar"].get(symbol)

    def set_last_bar(self, symbol: str, ts: str) -> None:
        self.state["last_bar"][symbol] = ts
        self.save_state()

    def in_cooldown(self, symbol: str) -> bool:
        until = self.state["cooldowns"].get(symbol)
        return bool(until) and _utcnow() < datetime.fromisoformat(until)

    # ---------------------------------------------------------------- account
    @retry_api
    def account(self) -> AccountInfo:
        a = self.api.get_account()
        return AccountInfo(
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            non_marginable_buying_power=float(getattr(a, "non_marginable_buying_power", a.cash)),
            cash=float(a.cash),
        )

    @retry_api
    def _clock(self):
        return self.api.get_clock()

    def market_open(self) -> bool:
        """US equity market status (cached 20s). Crypto never calls this."""
        ts, val = self._clock_cache
        if time.time() - ts < 20:
            return val
        val = bool(self._clock().is_open)
        self._clock_cache = (time.time(), val)
        return val

    @retry_api
    def broker_positions(self) -> Dict[str, object]:
        return {p.symbol: p for p in self.api.list_positions()}

    def gross_exposure(self) -> float:
        return sum(abs(float(p.market_value)) for p in self.broker_positions().values())

    @retry_api
    def _broker_position(self, inst):
        try:
            return self.api.get_position(inst.broker_symbol)
        except APIError as exc:
            if status_of(exc) == 404:
                return None
            raise

    # ----------------------------------------------------------------- orders
    @retry_api
    def _submit(self, **kwargs):
        return self.api.submit_order(**kwargs)

    @retry_api
    def _get_order(self, order_id: str):
        return self.api.get_order(order_id)

    @retry_api
    def _cancel(self, order_id: str) -> None:
        try:
            self.api.cancel_order(order_id)
        except APIError as exc:
            if status_of(exc) in (404, 422):       # gone / already filled or cancelled
                return
            raise

    def _wait_for_terminal(self, order_id: str, timeout: float):
        deadline = time.time() + timeout
        order = self._get_order(order_id)
        while order.status not in TERMINAL and time.time() < deadline:
            time.sleep(1.0)
            order = self._get_order(order_id)
        if order.status not in TERMINAL:
            log.warning("order %s still %s after %.0fs; cancelling", order_id, order.status, timeout)
            self._cancel(order_id)
            time.sleep(1.0)
            order = self._get_order(order_id)
        return order

    def _cancel_and_confirm(self, order_id: str):
        self._cancel(order_id)
        deadline = time.time() + 10
        order = self._get_order(order_id)
        while order.status not in TERMINAL and time.time() < deadline:
            time.sleep(0.5)
            order = self._get_order(order_id)
        return order

    # ------------------------------------------------------------ hard stop
    def _place_stop(self, pos: TrackedPosition) -> str:
        inst = self.instruments[pos.symbol]
        side = "sell" if pos.direction == "long" else "buy"
        if inst.asset_class == config.CRYPTO:
            buf = config.CRYPTO_STOP_LIMIT_BUFFER_PCT
            limit = pos.hard_stop_price * (1 - buf if pos.direction == "long" else 1 + buf)
            order = self._submit(symbol=inst.symbol, qty=round(pos.qty, 9), side=side,
                                 type="stop_limit", time_in_force="gtc",
                                 stop_price=_px(pos.hard_stop_price), limit_price=_px(limit))
        else:
            order = self._submit(symbol=inst.symbol, qty=int(pos.qty), side=side,
                                 type="stop", time_in_force="gtc",
                                 stop_price=_px(pos.hard_stop_price))
        log.info("hard stop placed %s %s @ %s (order %s)", pos.symbol, side,
                 _px(pos.hard_stop_price), order.id)
        return order.id

    def _attach_stop_or_flatten(self, pos: TrackedPosition) -> bool:
        """The hard stop is mandatory: if it cannot be placed, flatten at once."""
        try:
            pos.stop_order_id = self._place_stop(pos)
            return True
        except Exception as exc:  # noqa: BLE001
            log.critical("could not place hard stop for %s (%s) -> flattening position",
                         pos.symbol, exc)
            self.close_position(pos.symbol, reason="hard stop could not be placed")
            return False

    # ------------------------------------------------------------- open/close
    def open_position(self, inst, direction: str, qty: float, risk_dollars: float,
                      atr: float, strategy: str, trailing_mult: Optional[float]
                      ) -> Optional[TrackedPosition]:
        side = "buy" if direction == "long" else "sell"
        tif = "gtc" if inst.asset_class == config.CRYPTO else "day"
        log.info("ENTRY %s %s qty=%s", direction.upper(), inst.symbol, qty)
        order = self._submit(symbol=inst.symbol, qty=qty, side=side,
                             type="market", time_in_force=tif)
        order = self._wait_for_terminal(order.id, config.ORDER_FILL_TIMEOUT_S)
        filled = float(order.filled_qty or 0)
        if filled <= 0:
            log.warning("entry order for %s not filled (status=%s)", inst.symbol, order.status)
            return None

        entry = float(order.filled_avg_price)
        held = self._held_qty(inst, fallback=filled)
        pos = TrackedPosition(
            symbol=inst.symbol, direction=direction, qty=held, entry_price=entry,
            entry_time=_utcnow().isoformat(), strategy=strategy, atr_at_entry=float(atr),
            hard_stop_price=hard_stop_price(direction, entry, held, risk_dollars),
            trailing_mult=trailing_mult, extreme_price=entry, last_atr=float(atr),
        )
        if trailing_mult:
            gap = trailing_mult * atr
            pos.trailing_stop = entry - gap if direction == "long" else entry + gap
        self.positions[inst.symbol] = pos
        self.save_state()
        log.info("FILLED %s %s qty=%s @ %.4f | hard stop %.2f | trailing %s",
                 direction.upper(), inst.symbol, held, entry, pos.hard_stop_price,
                 f"{pos.trailing_stop:.2f}" if pos.trailing_stop else "-")
        if self._attach_stop_or_flatten(pos):
            self.save_state()
            return pos
        return None

    def _held_qty(self, inst, fallback: float) -> float:
        """Broker-reported position size (crypto fees are taken in kind)."""
        if inst.asset_class != config.CRYPTO:
            return float(fallback)
        for _ in range(5):
            bp = self._broker_position(inst)
            if bp is not None:
                return abs(float(bp.qty))
            time.sleep(1.0)
        return float(fallback)

    def close_position(self, symbol: str, reason: str, price_hint: Optional[float] = None
                       ) -> Optional[dict]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        inst = self.instruments[symbol]
        log.info("EXIT %s %s (%s)", pos.direction.upper(), symbol, reason)

        # Cancel the protective stop first: Alpaca rejects an opposing order while
        # the stop is holding the shares (wash-trade / insufficient-qty guard).
        if pos.stop_order_id:
            stop = self._cancel_and_confirm(pos.stop_order_id)
            if stop.status == "filled":      # stop fired just before we got here
                return self._finalize(pos, float(stop.filled_avg_price),
                                      "hard_stop (broker)", stop.filled_at)
            pos.stop_order_id = None

        try:
            closing = self._close_at_broker(inst)
        except APIError as exc:
            if status_of(exc) == 404:        # nothing left at the broker
                price = price_hint or self.md.last_price(inst)
                return self._finalize(pos, price, f"{reason} (already flat at broker)", None)
            raise
        order = self._wait_for_terminal(closing.id, config.ORDER_FILL_TIMEOUT_S)
        if float(order.filled_qty or 0) <= 0:
            log.error("close order for %s did not fill (status=%s); re-arming hard stop",
                      symbol, order.status)
            self._attach_stop_or_flatten_no_recurse(pos)
            return None
        return self._finalize(pos, float(order.filled_avg_price), reason, order.filled_at)

    @retry_api
    def _close_at_broker(self, inst):
        return self.api.close_position(inst.broker_symbol)

    def _attach_stop_or_flatten_no_recurse(self, pos: TrackedPosition) -> None:
        try:
            pos.stop_order_id = self._place_stop(pos)
            self.save_state()
        except Exception as exc:  # noqa: BLE001
            log.critical("could not re-arm hard stop for %s: %s -- software stop still active",
                         pos.symbol, exc)

    # --------------------------------------------------------------- bookkeeping
    def _finalize(self, pos: TrackedPosition, exit_price: float, reason: str,
                  exit_time) -> dict:
        sign = 1 if pos.direction == "long" else -1
        pnl = sign * (exit_price - pos.entry_price) * pos.qty
        when = _to_dt(exit_time) or _utcnow()
        record = {
            "timestamp": when.isoformat(),
            "instrument": pos.symbol,
            "direction": pos.direction,
            "entry_price": round(pos.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "profit_loss": round(pnl, 2),
            "position_size": pos.qty,
        }
        self._append_csv(self.trades_csv, TRADE_FIELDS, record)
        day = self.state.setdefault("day", {})
        if day:
            day["realized"] = round(day.get("realized", 0.0) + pnl, 2)
            day["trades"] = day.get("trades", 0) + 1
        self.positions.pop(pos.symbol, None)
        if "stop" in reason and self.cooldown_seconds.get(pos.symbol):
            until = _utcnow().timestamp() + self.cooldown_seconds[pos.symbol]
            self.state["cooldowns"][pos.symbol] = datetime.fromtimestamp(until, timezone.utc).isoformat()
        self.save_state()
        log.info("CLOSED %s %s qty=%s entry=%.4f exit=%.4f P&L=%+.2f (%s)",
                 pos.direction.upper(), pos.symbol, pos.qty, pos.entry_price,
                 exit_price, pnl, reason)
        return record

    def poll_stop_fills(self) -> None:
        """Detect broker-side stop fills and (re)arm missing stops."""
        for sym, pos in list(self.positions.items()):
            try:
                if not pos.stop_order_id:
                    self._attach_stop_or_flatten(pos)
                    continue
                order = self._get_order(pos.stop_order_id)
                if order.status == "filled":
                    self._finalize(pos, float(order.filled_avg_price),
                                   "hard_stop (broker)", order.filled_at)
                elif order.status in ("canceled", "expired", "rejected"):
                    log.warning("stop order for %s is %s; re-arming", sym, order.status)
                    pos.stop_order_id = None
                    self._attach_stop_or_flatten(pos)
            except Exception as exc:  # noqa: BLE001
                log.error("stop poll failed for %s: %s", sym, exc)

    # ------------------------------------------------------------ reconciliation
    def reconcile(self) -> Dict[str, object]:
        """Compare tracked positions with the broker. Returns broker positions
        for instruments the bot trades but does not yet track (to be adopted)."""
        broker = self.broker_positions()
        for sym, pos in list(self.positions.items()):
            inst = self.instruments[sym]
            bp = broker.get(inst.broker_symbol)
            if bp is None:
                price, reason = self._infer_exit(pos, inst)
                log.warning("%s no longer held at broker -> recording close (%s)", sym, reason)
                self._finalize(pos, price, reason, None)
            else:
                held = abs(float(bp.qty))
                if abs(held - pos.qty) > 1e-9:
                    log.warning("%s qty differs (tracked %s, broker %s); using broker",
                                sym, pos.qty, held)
                    pos.qty = held
                    self.save_state()
        return {inst.symbol: broker[inst.broker_symbol]
                for inst in self.instruments.values()
                if inst.broker_symbol in broker and inst.symbol not in self.positions}

    def _infer_exit(self, pos: TrackedPosition, inst):
        if pos.stop_order_id:
            try:
                o = self._get_order(pos.stop_order_id)
                if o.status == "filled":
                    return float(o.filled_avg_price), "hard_stop (broker)"
            except Exception:  # noqa: BLE001
                pass
        return self.md.last_price(inst), "closed externally"

    def adopt_position(self, inst, broker_pos, atr: float, strategy: str,
                       trailing_mult: Optional[float], risk_dollars: float
                       ) -> Optional[TrackedPosition]:
        """Take over a position the bot did not open: attach the mandatory hard stop."""
        qty = abs(float(broker_pos.qty))
        direction = "long" if float(broker_pos.qty) > 0 else "short"
        entry = float(broker_pos.avg_entry_price)
        pos = TrackedPosition(
            symbol=inst.symbol, direction=direction, qty=qty, entry_price=entry,
            entry_time=_utcnow().isoformat(), strategy=strategy, atr_at_entry=float(atr),
            hard_stop_price=hard_stop_price(direction, entry, qty, risk_dollars),
            trailing_mult=trailing_mult, extreme_price=entry, last_atr=float(atr),
        )
        if trailing_mult:
            gap = trailing_mult * atr
            pos.trailing_stop = entry - gap if direction == "long" else entry + gap
        self.positions[inst.symbol] = pos
        self.save_state()
        log.warning("ADOPTED untracked %s position in %s qty=%s entry=%.4f",
                    direction, inst.symbol, qty, entry)
        # Existing open orders (e.g. a manual stop) would block ours; the stop
        # placement below will surface any such conflict loudly.
        self._attach_stop_or_flatten(pos)
        return self.positions.get(inst.symbol)

    # ----------------------------------------------------------------- CSV I/O
    @staticmethod
    def _append_csv(path, header: List[str], row: dict) -> None:
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def _upsert_daily_row(self, day: dict, ending_equity: float) -> None:
        start = float(day["start_equity"])
        row = {
            "date": day["date"],
            "starting_equity": round(start, 2),
            "ending_equity": round(ending_equity, 2),
            "pnl": round(ending_equity - start, 2),
            "pnl_pct": round((ending_equity - start) / start * 100, 4) if start else 0.0,
            "realized_pnl": round(day.get("realized", 0.0), 2),
            "trades": day.get("trades", 0),
        }
        rows: Dict[str, dict] = {}
        if os.path.exists(self.pnl_csv):
            with open(self.pnl_csv, newline="") as fh:
                rows = {r["date"]: r for r in csv.DictReader(fh)}
        rows[row["date"]] = row
        tmp = f"{self.pnl_csv}.tmp"
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PNL_FIELDS)
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])
        os.replace(tmp, self.pnl_csv)

    def update_daily_pnl(self, equity: float, now: Optional[datetime] = None,
                         force: bool = False) -> None:
        """Maintain daily_pnl.csv. The 'day' is the America/New_York calendar day."""
        now = now or _utcnow()
        today = now.astimezone(ZoneInfo(config.MARKET_TZ)).date().isoformat()
        day = self.state.get("day") or {}
        if not day:
            self.state["day"] = {"date": today, "start_equity": equity, "realized": 0.0,
                                 "trades": 0, "last_equity": equity}
            self.save_state()
            return
        if day["date"] != today:
            # Close out the previous day using the last equity we saw during it.
            self._upsert_daily_row(day, float(day.get("last_equity", equity)))
            self.state["day"] = {"date": today, "start_equity": equity, "realized": 0.0,
                                 "trades": 0, "last_equity": equity}
            self.save_state()
            self._last_pnl_write = time.time()
            return
        day["last_equity"] = equity
        if force or time.time() - self._last_pnl_write > 300:
            self._upsert_daily_row(day, equity)
            self.save_state()
            self._last_pnl_write = time.time()


def _to_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

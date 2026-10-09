"""Status snapshot the bot publishes for the monitor page (read-only, best effort).

The trading loop calls `build_snapshot` + `write_snapshot` at the end of every risk tick.
Nothing here may raise into the trading path: callers wrap it in try/except, and the
monitor treats a missing or old file as "bot down".
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional


def _pct(numer: float, denom: float) -> Optional[float]:
    return round(numer / denom * 100, 3) if denom else None


def position_view(pos, price: Optional[float], sma: Optional[float] = None) -> dict:
    """One open position as the monitor shows it. `price`/`sma` may be unknown (None)."""
    sign = 1 if pos.direction == "long" else -1
    view = {
        "symbol": pos.symbol, "direction": pos.direction, "qty": pos.qty,
        "entry_price": pos.entry_price, "entry_time": pos.entry_time, "strategy": pos.strategy,
        "hard_stop": pos.hard_stop_price, "trailing_stop": pos.trailing_stop,
        "stop_order_id": pos.stop_order_id, "price": price, "sma": sma,
        "unrealized": None, "unrealized_pct": None, "to_stop_pct": None, "to_sma_pct": None,
    }
    if price:
        view["unrealized"] = round(sign * (price - pos.entry_price) * pos.qty, 2)
        view["unrealized_pct"] = _pct(sign * (price - pos.entry_price), pos.entry_price)
        view["to_stop_pct"] = _pct(sign * (price - pos.hard_stop_price), price)
        if sma:
            view["to_sma_pct"] = _pct(sign * (price - sma), price)
    return view


def build_snapshot(*, started_at: str, mode: str, market_open: Optional[bool], equity: float,
                   cash: float, positions: list, signals: Dict[str, dict],
                   now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {
        "updated_at": now.isoformat(), "started_at": started_at, "pid": os.getpid(),
        "mode": mode, "market_open": market_open, "equity": equity, "cash": cash,
        "positions": positions, "signals": signals,
    }


def write_snapshot(path, snapshot: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(snapshot, fh, indent=2, default=str)
    os.replace(tmp, path)

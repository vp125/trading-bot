"""Risk management: ATR position sizing, hard stop, trailing stop, correlation filter.

Sizing rule
-----------
    risk_dollars = equity * RISK_PER_TRADE_PCT                 (1% of equity)
    qty          = risk_dollars / ATR(14)                      (1 ATR move == 1% of equity)

Quiet instruments (small ATR) therefore get bigger positions, volatile ones
smaller, so each position carries the same dollar risk per ATR.

Hard stop
---------
The stop is placed so that, at the stop price, the loss on the *actual* position
equals exactly `risk_dollars` (1% of equity):

    stop_distance = risk_dollars / qty

When the position is sized purely by ATR this is exactly 1 ATR from entry.
If a notional cap reduces the quantity, the stop distance widens in proportion
so the maximum loss is still 1% of equity (never more).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def hard_stop_price(direction: str, entry: float, qty: float, risk_dollars: float) -> float:
    """Stop price at which the loss on `qty` units equals `risk_dollars`.

    Rounded to the cent in the direction that makes the stop *tighter*, so
    rounding can only reduce the loss, never increase it.
    """
    distance = risk_dollars / qty
    if direction == "long":
        return math.ceil((entry - distance) * 100) / 100
    return math.floor((entry + distance) * 100) / 100


def _floor(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor + 1e-9) / factor


@dataclass
class SizeResult:
    qty: float
    risk_dollars: float
    notional: float
    stop_distance: float
    capped_by: Optional[str]      # None | "position_cap" | "gross_cap" | "buying_power"


class RiskManager:
    # ---------------------------------------------------------------- sizing
    def position_size(self, instrument, price: float, atr: float, equity: float,
                      buying_power: float, current_gross: float,
                      stop_atr_mult: float = 1.0) -> Optional[SizeResult]:
        """`stop_atr_mult` widens the hard stop to that many ATRs; the quantity shrinks
        in proportion so the loss at the stop is still RISK_PER_TRADE_PCT of equity."""
        if not all(map(math.isfinite, (price, atr, equity, buying_power, stop_atr_mult))):
            return None
        if price <= 0 or atr <= 0 or equity <= 0 or stop_atr_mult <= 0:
            return None

        risk_dollars = equity * config.RISK_PER_TRADE_PCT
        qty = risk_dollars / (atr * stop_atr_mult)    # stop_atr_mult ATR move == risk_dollars
        capped_by = None

        limits = {
            "position_cap": equity * config.MAX_POSITION_NOTIONAL_PCT,
            "gross_cap": max(0.0, equity * config.MAX_GROSS_EXPOSURE_PCT - current_gross),
            "buying_power": max(0.0, buying_power * 0.95),   # slack for price movement
        }
        name, limit = min(limits.items(), key=lambda kv: kv[1])
        if qty * price > limit:
            qty = limit / price
            capped_by = name

        # Equities trade in whole shares (also required for short sales);
        # crypto is fractional.
        qty = math.floor(qty) if instrument.asset_class == config.EQUITY else _floor(qty, 6)
        notional = qty * price
        if qty <= 0 or notional < config.MIN_ORDER_NOTIONAL:
            return None

        distance = risk_dollars / qty
        if distance >= price:      # a long stop would be at/below zero: position too small to matter
            return None
        return SizeResult(qty=qty, risk_dollars=risk_dollars, notional=notional,
                          stop_distance=distance, capped_by=capped_by)

    # ----------------------------------------------------------- correlation
    @staticmethod
    def correlation_allows(symbol: str, direction: str, positions: Dict[str, object]) -> Tuple[bool, str]:
        """Block e.g. a BTC/USD long while SPY *and* QQQ are both long."""
        for rule in config.CORRELATION_RULES:
            if rule["blocked"] != symbol or rule["direction"] != direction:
                continue
            if all(getattr(positions.get(s), "direction", None) == direction
                   for s in rule["if_long"]):
                return False, (f"correlation filter: {' & '.join(rule['if_long'])} already "
                               f"{direction}, no new {direction} on {symbol}")
        return True, ""

    # ------------------------------------------------------------- stop logic
    @staticmethod
    def hard_stop_hit(pos, price: float) -> bool:
        if pos.direction == "long":
            return price <= pos.hard_stop_price
        return price >= pos.hard_stop_price

    @staticmethod
    def update_trailing_stop(pos, price: float, atr: Optional[float] = None) -> None:
        """Ratchet the ATR trailing stop. It only ever moves in the trade's favour."""
        if not pos.trailing_mult:
            return
        if atr and math.isfinite(atr) and atr > 0:
            pos.last_atr = float(atr)
        if pos.last_atr <= 0:
            return
        gap = pos.trailing_mult * pos.last_atr
        if pos.direction == "long":
            pos.extreme_price = max(pos.extreme_price, price)
            candidate = pos.extreme_price - gap
            pos.trailing_stop = candidate if pos.trailing_stop is None \
                else max(pos.trailing_stop, candidate)
        else:
            pos.extreme_price = min(pos.extreme_price, price)
            candidate = pos.extreme_price + gap
            pos.trailing_stop = candidate if pos.trailing_stop is None \
                else min(pos.trailing_stop, candidate)

    @staticmethod
    def trailing_stop_hit(pos, price: float) -> bool:
        if pos.trailing_stop is None:
            return False
        if pos.direction == "long":
            return price <= pos.trailing_stop
        return price >= pos.trailing_stop

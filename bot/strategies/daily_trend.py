"""Strategy 4 - Daily trend filter (SPY, QQQ, GLD), long-only.

Entry : daily close above the 200-day SMA while flat -> long.
Exit  : daily close below the 200-day SMA -> close the long.
Both rules are level-based, so a missed bar or a restart cannot leave the bot on
the wrong side of the trend. The strategy never goes short; protection beyond the
signal is the broker-side hard stop, sized at `stop_atr_mult` x ATR (see
`Strategy.stop_atr_mult` and `RiskManager.position_size`).
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..indicators import sma
from .base import Signal, Strategy


class DailyTrendStrategy(Strategy):
    name = "daily_trend"
    trailing_atr_mult = None

    def evaluate(self, symbol: str, bars: pd.DataFrame,
                 direction: Optional[str]) -> Signal:
        period = int(self.params["sma_period"])
        min_bars = int(self.params.get("min_bars", period + 1))
        if len(bars) < min_bars:
            return Signal(reason=f"need {min_bars} daily bars for the {period}-day SMA, have {len(bars)}")

        close = float(bars["close"].iloc[-1])
        avg = float(sma(bars["close"], period).iloc[-1])
        if math.isnan(close) or math.isnan(avg):
            return Signal(reason="SMA not ready")
        info = {"close": close, "sma": avg}

        if direction == "long":
            if close < avg:
                return Signal(exit=True, reason=f"close {close:.2f} below {period}d SMA {avg:.2f}", info=info)
            return Signal(reason="holding long", info=info)
        if direction is None and close > avg:
            return Signal(enter="long", reason=f"close {close:.2f} above {period}d SMA {avg:.2f}", info=info)
        return Signal(reason="below SMA, flat" if direction is None else f"unexpected {direction} position",
                      info=info)

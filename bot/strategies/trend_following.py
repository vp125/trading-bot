"""Strategy 3 - Trend following (GLD, USO) on 4-hour candles.

Entry : 50 EMA crosses ABOVE the 200 EMA  -> long
        50 EMA crosses BELOW the 200 EMA  -> short (if enabled in config,
        otherwise the cross only exits longs).
Exit  : a long is closed whenever fast < slow, a short whenever fast > slow.
        Exits are level-based (not edge-based) so that a missed bar or a
        restart cannot leave the bot holding the wrong side of the trend.
        Entries remain edge-based: the bot only enters on the cross itself.
Trailing stop (3x ATR) is applied by the risk manager, see
`trailing_atr_mult`.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..indicators import ema
from .base import Signal, Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    @property
    def trailing_atr_mult(self) -> float:  # type: ignore[override]
        return float(self.params["trailing_atr_mult"])

    def evaluate(self, symbol: str, bars: pd.DataFrame,
                 direction: Optional[str]) -> Signal:
        fast_n = int(self.params["fast_ema"])
        slow_n = int(self.params["slow_ema"])
        min_bars = int(self.params.get("min_bars", slow_n + 1))
        go_short = bool(self.params.get("go_short_on_cross_down", True))
        if len(bars) < min_bars:
            return Signal(reason=f"need {min_bars} bars for EMA warm-up, have {len(bars)}")

        close = bars["close"]
        fast = ema(close, fast_n)
        slow = ema(close, slow_n)
        f1, f0 = fast.iloc[-1], fast.iloc[-2]
        s1, s0 = slow.iloc[-1], slow.iloc[-2]
        if any(math.isnan(v) for v in (f1, f0, s1, s0)):
            return Signal(reason="EMAs not ready")

        info = {"fast": float(f1), "slow": float(s1)}
        crossed_up = f0 <= s0 and f1 > s1
        crossed_down = f0 >= s0 and f1 < s1

        if direction == "long":
            if f1 < s1:
                return Signal(exit=True,
                              enter="short" if (crossed_down and go_short) else None,
                              reason="50 EMA below 200 EMA", info=info)
            return Signal(reason="holding long", info=info)
        if direction == "short":
            if f1 > s1:
                return Signal(exit=True,
                              enter="long" if crossed_up else None,
                              reason="50 EMA above 200 EMA", info=info)
            return Signal(reason="holding short", info=info)

        if crossed_up:
            return Signal(enter="long", reason="50 EMA crossed above 200 EMA", info=info)
        if crossed_down and go_short:
            return Signal(enter="short", reason="50 EMA crossed below 200 EMA", info=info)
        return Signal(reason="no cross", info=info)

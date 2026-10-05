"""Strategy 1 - Mean reversion (SPY, QQQ) on 15-minute candles.

Entry : close more than k standard deviations below the 20-period SMA -> long,
        more than k above -> short.   (k = 1.5 SPY, 1.8 QQQ)
Exit  : close returns to the moving average (long: close >= SMA,
        short: close <= SMA). The exit is level-based, so a missed bar
        cannot leave a position stranded.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..indicators import rolling_std, sma
from .base import Signal, Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    trailing_atr_mult = None

    def evaluate(self, symbol: str, bars: pd.DataFrame,
                 direction: Optional[str]) -> Signal:
        period = int(self.params["period"])
        k = float(self.params["std_mult"][symbol])
        if len(bars) < period:
            return Signal(reason=f"need {period} bars, have {len(bars)}")

        close = bars["close"]
        mean = sma(close, period).iloc[-1]
        std = rolling_std(close, period).iloc[-1]
        px = close.iloc[-1]
        if any(math.isnan(v) for v in (mean, std, px)) or std <= 0:
            return Signal(reason="indicator not ready / zero volatility")

        z = (px - mean) / std
        info = {"price": float(px), "sma": float(mean), "std": float(std), "z": float(z)}

        if direction == "long":
            if px >= mean:
                return Signal(exit=True, reason="mean reached (long exit)", info=info)
            return Signal(reason="holding long", info=info)
        if direction == "short":
            if px <= mean:
                return Signal(exit=True, reason="mean reached (short exit)", info=info)
            return Signal(reason="holding short", info=info)

        if z < -k:
            return Signal(enter="long", reason=f"z={z:.2f} < -{k}", info=info)
        if z > k:
            return Signal(enter="short", reason=f"z={z:.2f} > {k}", info=info)
        return Signal(reason=f"z={z:.2f} inside +/-{k}", info=info)

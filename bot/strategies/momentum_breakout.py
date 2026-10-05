"""Strategy 2 - Momentum breakout (BTC/USD) on 1-hour candles.

Channel : highest high / lowest low of the PREVIOUS 20 completed candles
          (the signal candle itself is excluded, otherwise it could never break
          its own range).
Volume  : signal candle volume must be >= 1.5x the average volume of those
          same 20 candles.
Long    : candle closes above the channel high on confirmed volume.
Short   : candle closes below the channel low on confirmed volume ->
          exit an open long; open a short only if the instrument allows it
          (Alpaca does not support crypto shorts, so for BTC/USD this is
          exit-only; the executor drops the short entry).
Trailing stop (2x ATR) is applied by the risk manager, see
`trailing_atr_mult`.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from .base import Signal, Strategy


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"

    @property
    def trailing_atr_mult(self) -> float:  # type: ignore[override]
        return float(self.params["trailing_atr_mult"])

    def evaluate(self, symbol: str, bars: pd.DataFrame,
                 direction: Optional[str]) -> Signal:
        n = int(self.params["period"])
        vol_mult = float(self.params["volume_mult"])
        if len(bars) < n + 1:
            return Signal(reason=f"need {n + 1} bars, have {len(bars)}")

        prior = bars.iloc[-(n + 1):-1]
        last = bars.iloc[-1]
        high_n = float(prior["high"].max())
        low_n = float(prior["low"].min())
        avg_vol = float(prior["volume"].mean())
        if math.isnan(avg_vol) or avg_vol <= 0:
            return Signal(reason="no volume history")

        vol_ratio = float(last["volume"]) / avg_vol
        vol_ok = vol_ratio >= vol_mult
        close = float(last["close"])
        info = {"close": close, "high_n": high_n, "low_n": low_n,
                "volume_ratio": vol_ratio}

        if close > high_n and vol_ok:
            reason = f"close {close:.2f} > {n}-bar high {high_n:.2f}, vol x{vol_ratio:.2f}"
            if direction == "long":
                return Signal(reason="already long", info=info)
            return Signal(exit=direction == "short", enter="long",
                          reason=reason, info=info)

        if close < low_n and vol_ok:
            reason = f"close {close:.2f} < {n}-bar low {low_n:.2f}, vol x{vol_ratio:.2f}"
            if direction == "short":
                return Signal(reason="already short", info=info)
            return Signal(exit=direction == "long", enter="short",
                          reason=reason, info=info)

        why = "no breakout" if close <= high_n and close >= low_n else \
            f"breakout without volume (x{vol_ratio:.2f} < x{vol_mult})"
        return Signal(reason=why, info=info)

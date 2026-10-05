"""Strategy interface.

A strategy is a pure function of (symbol, completed bars, current position
direction) -> Signal. It never touches the broker, sizes positions or places
stops; that is the job of the risk manager / portfolio. This keeps strategies
trivially testable on synthetic data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pandas as pd


@dataclass
class Signal:
    exit: bool = False                 # close the current position (if any)
    enter: Optional[str] = None        # "long" | "short" | None  (after any exit)
    reason: str = ""
    info: dict = field(default_factory=dict)

    @property
    def is_hold(self) -> bool:
        return not self.exit and self.enter is None


class Strategy(ABC):
    name: str = "base"
    # ATR multiple for the trailing stop; None means the strategy has none.
    trailing_atr_mult: Optional[float] = None

    def __init__(self, params: dict):
        self.params = params

    @property
    def timeframe(self) -> Tuple[int, str]:
        return tuple(self.params["timeframe"])  # type: ignore[return-value]

    @property
    def stop_atr_mult(self) -> float:
        """Hard-stop distance in ATRs; sizing makes that distance cost 1% of equity."""
        return float(self.params.get("stop_atr_mult", 1.0))

    @property
    def check_interval_s(self) -> int:
        return int(self.params["check_interval_s"])

    @property
    def lookback_days(self) -> int:
        return int(self.params["lookback_days"])

    @abstractmethod
    def evaluate(self, symbol: str, bars: pd.DataFrame,
                 direction: Optional[str]) -> Signal:
        """`bars`: completed candles, oldest first, columns open/high/low/close/volume.
        `direction`: "long", "short" or None (flat) for this symbol."""

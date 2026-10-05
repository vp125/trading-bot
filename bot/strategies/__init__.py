from .base import Signal, Strategy
from .daily_trend import DailyTrendStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .trend_following import TrendFollowingStrategy

STRATEGY_CLASSES = {
    "daily_trend": DailyTrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
}

__all__ = [
    "Signal",
    "Strategy",
    "DailyTrendStrategy",
    "MeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "TrendFollowingStrategy",
    "STRATEGY_CLASSES",
]

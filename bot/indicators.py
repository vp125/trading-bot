"""Pure indicator functions (pandas in, pandas out). No I/O."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Population standard deviation (ddof=0), as used by Bollinger Bands."""
    return series.rolling(period, min_periods=period).std(ddof=0)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA of true range)."""
    tr = true_range(df)
    # first bar has no previous close -> its TR is just high-low; drop it from the seed
    tr.iloc[0] = float("nan")
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

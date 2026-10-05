"""Market-data access: completed candles and latest trade price."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import config
from .api_utils import retry_api

log = logging.getLogger(__name__)

_UNITS = {"Minute": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}
_SECONDS = {"Minute": 60, "Hour": 3600, "Day": 86400}
OHLCV = ["open", "high", "low", "close", "volume"]


def to_timeframe(spec: Tuple[int, str]) -> TimeFrame:
    amount, unit = spec
    return TimeFrame(amount, _UNITS[unit])


def bar_seconds(spec: Tuple[int, str]) -> int:
    amount, unit = spec
    return amount * _SECONDS[unit]


def regular_hours_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep candles that START inside the 09:30-16:00 America/New_York session."""
    local = df.index.tz_convert(config.MARKET_TZ)
    minutes = local.hour * 60 + local.minute
    keep = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60) & (local.dayofweek < 5)
    return df[keep]


def clean_bars(df: pd.DataFrame, instrument, spec: Tuple[int, str],
               now: datetime) -> pd.DataFrame:
    """Normalise raw API candles: OHLCV floats, sorted/unique, regular-session
    filter for intraday equity bars, and the still-forming candle dropped."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if (config.RTH_ONLY_FOR_MINUTE_BARS and instrument.asset_class == config.EQUITY
            and spec[1] == "Minute"):
        df = regular_hours_only(df)
        if df.empty:
            return df
    if df.index[-1] + timedelta(seconds=bar_seconds(spec)) > now:
        df = df.iloc[:-1]
    return df


class MarketData:
    def __init__(self, api):
        self.api = api

    @retry_api
    def _fetch(self, instrument, spec: Tuple[int, str], start_iso: str,
               end_iso: Optional[str] = None) -> pd.DataFrame:
        tf = to_timeframe(spec)
        if instrument.asset_class == config.CRYPTO:
            bars = self.api.get_crypto_bars(instrument.symbol, tf, start=start_iso, end=end_iso)
        else:
            bars = self.api.get_bars(instrument.symbol, tf, start=start_iso, end=end_iso,
                                     feed=config.DATA_FEED)
        return bars.df

    def get_bars(self, instrument, spec: Tuple[int, str], lookback_days: int,
                 now: Optional[datetime] = None) -> pd.DataFrame:
        """Completed candles only, oldest first. The still-forming candle is dropped."""
        now = now or datetime.now(timezone.utc)
        start = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return clean_bars(self._fetch(instrument, spec, start), instrument, spec, now)

    def get_bars_between(self, instrument, spec: Tuple[int, str], start: datetime,
                         end: Optional[datetime] = None) -> pd.DataFrame:
        """Historical candles for [start, end] (used by the backtester)."""
        end = end or datetime.now(timezone.utc)
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        raw = self._fetch(instrument, spec, start.strftime(fmt), end.strftime(fmt))
        return clean_bars(raw, instrument, spec, end)

    @retry_api
    def last_price(self, instrument) -> float:
        if instrument.asset_class == config.CRYPTO:
            trade = self.api.get_latest_crypto_trades([instrument.symbol])[instrument.symbol]
        else:
            trade = self.api.get_latest_trade(instrument.symbol, feed=config.DATA_FEED)
        return float(trade.price)

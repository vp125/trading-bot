"""Central configuration for the Alpaca multi-strategy bot.

Secrets come from `.env` (never hard-code keys). Everything else is a plain
constant here so strategy / risk parameters are easy to review and change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------------------------------- #
# Credentials / endpoints
# --------------------------------------------------------------------------- #
API_KEY = os.getenv("APCA_API_KEY_ID", "")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
# "iex" works on free data plans; use "sip" only if you pay for the full feed.
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
# Safety interlock: live endpoint requires an explicit opt-in.
CONFIRM_LIVE_TRADING = os.getenv("CONFIRM_LIVE_TRADING", "").lower() == "yes"


def is_paper() -> bool:
    return "paper-api" in BASE_URL


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
TRADES_CSV = BASE_DIR / "trades.csv"
DAILY_PNL_CSV = BASE_DIR / "daily_pnl.csv"
STATE_FILE = BASE_DIR / "state.json"      # open-position state survives restarts
LOG_FILE = BASE_DIR / "bot.log"

# --------------------------------------------------------------------------- #
# Instruments -> strategy mapping
# --------------------------------------------------------------------------- #
EQUITY = "equity"
CRYPTO = "crypto"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str            # EQUITY or CRYPTO
    strategy: str               # key into STRATEGY_PARAMS
    allow_short: bool           # can this instrument be sold short at all?

    @property
    def broker_symbol(self) -> str:
        """Symbol as the positions endpoint spells it (BTC/USD -> BTCUSD)."""
        return self.symbol.replace("/", "")


INSTRUMENTS: Dict[str, Instrument] = {
    # SPY/QQQ used to run 15-min mean reversion; it lost money in every backtest
    # window we tried (see README), so it was replaced by long-only daily trend.
    "SPY":     Instrument("SPY",     EQUITY, "daily_trend",       allow_short=False),
    "QQQ":     Instrument("QQQ",     EQUITY, "daily_trend",       allow_short=False),
    # BTC/USD (momentum_breakout) is disabled: it lost 45% with a 48% drawdown over a
    # 24-month backtest. To re-enable, restore this line. Alpaca does NOT support
    # shorting crypto, so a "break below the low" signal only exits an open long.
    # "BTC/USD": Instrument("BTC/USD", CRYPTO, "momentum_breakout", allow_short=False),
    "GLD":     Instrument("GLD",     EQUITY, "daily_trend",       allow_short=False),
    # Alpaca currently flags USO as hard-to-borrow / not shortable (checked via
    # the assets endpoint), so a cross-down on USO only exits longs. Flip this
    # to True if USO becomes shortable on your account.
    "USO":     Instrument("USO",     EQUITY, "trend_following",   allow_short=False),
}

# IEX returns thin pre/post-market candles (often 1-3 trades). For intraday
# (minute) equity bars keep only regular-session candles so they don't distort
# the 20-bar mean / standard deviation. Hourly+ bars are left as returned.
RTH_ONLY_FOR_MINUTE_BARS = True

# --------------------------------------------------------------------------- #
# Strategy parameters
# --------------------------------------------------------------------------- #
# Timeframes are (amount, unit) with unit in {"Minute", "Hour"}.
STRATEGY_PARAMS = {
    # Long-only: hold while the daily close is above its 200-day SMA, exit when it
    # closes below. No signal-based short. Positions are sized so that a
    # `stop_atr_mult` x ATR move (the hard stop) costs RISK_PER_TRADE_PCT of equity.
    "daily_trend": {
        "timeframe": (1, "Day"),
        "sma_period": 200,
        "stop_atr_mult": 3.0,
        # Daily bars are polled every 15 min during the session; a bar is acted on once.
        "check_interval_s": 15 * 60,
        "lookback_days": 420,               # ~290 trading days for the 200-day SMA
        "min_bars": 201,
    },
    # Not assigned to any instrument right now (kept for reference / tests).
    "mean_reversion": {
        "timeframe": (15, "Minute"),
        "period": 20,
        "std_mult": {"SPY": 1.5, "QQQ": 1.8},
        "check_interval_s": 15 * 60,
        "lookback_days": 10,
    },
    "momentum_breakout": {
        "timeframe": (1, "Hour"),
        "period": 20,
        "volume_mult": 1.5,
        "trailing_atr_mult": 2.0,
        "check_interval_s": 60 * 60,
        "lookback_days": 14,
    },
    "trend_following": {
        "timeframe": (4, "Hour"),
        "fast_ema": 50,
        "slow_ema": 200,
        "trailing_atr_mult": 3.0,
        "go_short_on_cross_down": True,     # False => only exit longs
        # 4h bars are polled every 15 min; a bar is acted on exactly once.
        "check_interval_s": 15 * 60,
        # EMA(200) needs a long warm-up to converge.
        "lookback_days": 400,
        "min_bars": 260,
    },
}

ATR_PERIOD = 14


def active_strategies() -> list:
    """Strategies that at least one instrument uses, in STRATEGY_PARAMS order."""
    used = {inst.strategy for inst in INSTRUMENTS.values()}
    return [name for name in STRATEGY_PARAMS if name in used]

# --------------------------------------------------------------------------- #
# Risk management
# --------------------------------------------------------------------------- #
# 1 ATR adverse move == this fraction of total account equity.
RISK_PER_TRADE_PCT = 0.01
# Hard stop: every trade is closed if its loss reaches RISK_PER_TRADE_PCT of
# equity. This is always on (no switch): it is placed as a broker-side stop
# order AND re-checked by the bot every RISK_TICK_SECONDS.
# Crypto only supports stop-limit; limit sits this far beyond the stop price.
CRYPTO_STOP_LIMIT_BUFFER_PCT = 0.01

# ATR sizing can imply huge notional when ATR is small relative to price
# (e.g. SPY on 15-min candles). These caps keep orders inside buying power.
# When a cap binds the trade risks LESS than 1% of equity, never more.
MAX_POSITION_NOTIONAL_PCT = 1.0     # per position, as multiple of equity
MAX_GROSS_EXPOSURE_PCT = 2.0        # all positions combined (Reg T overnight = 2x)

# Correlation filter: block new longs in `blocked` while all `if_long` are long.
CORRELATION_RULES = [
    {"if_long": ["SPY", "QQQ"], "blocked": "BTC/USD", "direction": "long"},
]

# After a hard-stop exit, wait this many strategy bars before re-entering.
COOLDOWN_BARS_AFTER_STOP = 2

# --------------------------------------------------------------------------- #
# Runtime behaviour
# --------------------------------------------------------------------------- #
RISK_TICK_SECONDS = 30              # trailing-stop / stop-fill monitoring cadence
ORDER_FILL_TIMEOUT_S = 45
API_MAX_RETRIES = 5
API_BACKOFF_BASE_S = 2.0
API_BACKOFF_MAX_S = 60.0
MARKET_TZ = "America/New_York"      # defines the "day" for daily_pnl.csv
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MIN_ORDER_NOTIONAL = 1.0            # Alpaca minimum order value (USD)


def validate() -> Optional[str]:
    """Return an error string if the configuration is unusable, else None."""
    if not API_KEY or not API_SECRET or "your_" in API_KEY.lower():
        return "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env"
    if not is_paper() and not CONFIRM_LIVE_TRADING:
        return ("APCA_API_BASE_URL points at LIVE trading. "
                "Set CONFIRM_LIVE_TRADING=yes in .env to proceed.")
    return None

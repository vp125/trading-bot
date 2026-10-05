import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import config  # noqa: E402


@pytest.fixture
def legacy_instruments(monkeypatch):
    """The pre-daily_trend mapping (SPY/QQQ mean reversion, GLD 4h trend) that the
    older engine/order-flow tests were written against. Those tests exercise generic
    machinery, so they pin this mapping instead of following config.INSTRUMENTS."""
    eq = config.EQUITY
    for sym, strat in (("SPY", "mean_reversion"), ("QQQ", "mean_reversion"),
                       ("GLD", "trend_following")):
        monkeypatch.setitem(config.INSTRUMENTS, sym,
                            config.Instrument(sym, eq, strat, allow_short=True))
    monkeypatch.setitem(config.INSTRUMENTS, "BTC/USD",
                        config.Instrument("BTC/USD", config.CRYPTO, "momentum_breakout",
                                          allow_short=False))

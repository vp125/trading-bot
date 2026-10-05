# AGENT.md

Guidance for AI coding agents working in this repo. See `README.md` for the full strategy and risk-rule write-up; this file covers what you need to change code safely.

## What this is

Alpaca multi-strategy trading bot (Python, REST only via `alpaca-trade-api`). Four instruments (SPY, QQQ, GLD, USO), two active strategies, one shared risk layer. Real money is possible, so treat every change as safety-relevant.

## Commands

Run from the repo root (`alpaca_bot/`):

```bash
py -3.12 -m venv .venv                # Python 3.12; 3.13 breaks alpaca-trade-api
.venv\Scripts\activate                # Windows (PowerShell/cmd)
pip install -r requirements.txt
python -m pytest tests -q            # offline, no network, fake broker
python -m bot.main                   # run the live/paper loop (needs keys in .env)
python -m bot.backtest --synthetic   # engine smoke test, no keys needed
python -m bot.backtest               # real 6-month backtest, needs keys
python -m bot.backtest --months 24   # better for the daily strategy (few trades per year)
```

Always run the tests after touching `bot/` or `config.py`. Never run `bot.main` or a real backtest unless the user asks; `bot.main` places orders.

## Architecture

- `config.py` — single source of truth: credentials (from `.env`), `INSTRUMENTS`, `STRATEGY_PARAMS`, risk constants. Add or tune parameters here, not inline.
- `bot/main.py` — `TradingBot`: scheduler (`Job`s), startup reconciliation, `_process_instrument` (strategy cycle) and `risk_tick` (30 s stop monitoring).
- `bot/strategies/` — pure `evaluate(symbol, bars, direction) -> Signal`. Strategies must never touch the broker, size positions or place stops. Register new ones in `strategies/__init__.py` (`STRATEGY_CLASSES`), add params to `STRATEGY_PARAMS`, and assign them to an instrument in `INSTRUMENTS`. Active now: `daily_trend` (SPY/QQQ/GLD, long-only, 200-day SMA, 3x ATR hard stop), `trend_following` (USO). `momentum_breakout` (BTC/USD, disabled: -45% / -48% DD over 24 months) and `mean_reversion` (lost money in all backtests) are kept but unassigned; `config.active_strategies()` decides which strategies get jobs and backtest rows.
- `bot/risk_manager.py` — ATR sizing (`stop_atr_mult` widens the hard stop and shrinks qty so loss stays 1%), caps, hard/trailing stops, correlation filter.
- `bot/portfolio.py` — orders, positions, `state.json`, `trades.csv`, `daily_pnl.csv`.
- `bot/data_feed.py`, `bot/indicators.py`, `bot/api_utils.py` — candles/prices, SMA/EMA/ATR, retry/backoff.
- `bot/backtest.py` — reuses the live strategy and risk code; keep it in sync when changing either.

## Rules that must not be broken

- **Paper by default.** Live endpoint requires `CONFIRM_LIVE_TRADING=yes` (`config.validate`). Do not weaken or bypass this.
- **Hard stop is always on**: broker-side stop order plus a software re-check. If the stop can't be placed after entry, the position is closed. Don't add a switch to disable it.
- **Risk per trade never exceeds 1% of equity**; caps may only reduce size (stop is widened so max loss stays 1%).
- **Signals use completed candles only**; each bar is acted on once (`last_bar` / `set_last_bar`). Backtest fills at the next candle's open, so no look-ahead.
- **No crypto shorts; USO not shortable; SPY/QQQ/GLD run long-only** — controlled by `allow_short` in `config.py`.
- One failing symbol or job must never kill the loop (existing `try/except` + retry pattern in `main.py`); keep that behavior.
- Keep `state.json` backward compatible so a restart can resume open positions.

## Conventions

- `from __future__ import annotations`, type hints, dataclasses, module docstrings explaining intent.
- Use the `logging` module (`log = logging.getLogger("bot")`), not `print`, inside `bot/`.
- Broker calls go through the retry helpers in `api_utils.py`.
- Tests live in `tests/` (pytest; `conftest.py` puts the repo root on `sys.path`). They use a fake broker, so new order-flow logic needs a test there. Tests verify logic, not profitability. `test_logic`, `test_backtest` and `test_end_to_end` pin the old SPY/QQQ-mean-reversion / GLD-trend mapping via the `legacy_instruments` fixture (they test generic machinery); `test_daily_trend.py` runs against the shipped config.
- Comments explain *why* (broker quirks, risk reasoning), matching the existing style.

## Secrets and generated files

- `.env` holds API keys and is git-ignored. Never print, log, commit or copy its values; don't read it unless necessary.
- Git-ignored runtime artifacts: `state.json`, `bot.log*`, `trades.csv`, `daily_pnl.csv`. Don't hand-edit `state.json` while the bot runs.
- `backtest_*` outputs and `backtest_data/` are generated; regenerate rather than edit.

## Environment notes

- Windows dev machine (PowerShell). `alpaca-trade-api` can fail to install on Python 3.13; see the README for the `--no-deps` workaround.
- Data defaults to the free IEX feed (`ALPACA_DATA_FEED=iex`).

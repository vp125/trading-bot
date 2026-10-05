# Alpaca multi-strategy trading bot

Trades four instruments at once, each with its own strategy, under one shared risk layer.
BTC/USD (momentum breakout) is currently disabled: it lost 45% with a 48% drawdown in a 24-month
backtest. Re-enable it by restoring its line in `INSTRUMENTS` in `config.py`.
SPY/QQQ used to run 15-min mean reversion; it lost money in every backtest window tried
(24 months, four 6-month windows, many parameter variants), so they now run a long-only daily trend filter.

| Instrument | Strategy | Candles | Entry | Exit |
|---|---|---|---|---|
| SPY, QQQ, GLD | Daily trend (long-only) | 1 day | close above the 200-day SMA | close below the 200-day SMA; hard stop 3x ATR |
| ~~BTC/USD~~ (disabled) | Momentum breakout | 1 hour | close beyond prior 20-bar high/low with volume >= 1.5x the 20-bar average | break of the opposite channel; 2x ATR trailing stop |
| USO | Trend following | 4 hour | 50 EMA crosses 200 EMA | cross back; 3x ATR trailing stop |

## Layout

```
alpaca_bot/
├── .env                        API keys (placeholders - fill in)
├── config.py                   all parameters
├── bot/
│   ├── main.py                 scheduler / loop / startup reconciliation
│   ├── portfolio.py            orders, positions, state.json, trades.csv, daily_pnl.csv
│   ├── risk_manager.py         ATR sizing, hard stop, trailing stop, correlation filter
│   ├── data_feed.py            candles + latest price
│   ├── backtest.py             6-month historical backtest + chart
│   ├── indicators.py           SMA / std / EMA / ATR
│   ├── api_utils.py            retry + backoff for disconnects / 429 / 5xx
│   └── strategies/
│       ├── daily_trend.py          SPY / QQQ / GLD (active)
│       ├── mean_reversion.py       not assigned to any instrument (kept for reference/tests)
│       ├── momentum_breakout.py
│       └── trend_following.py
└── tests/                      offline tests (strategies, sizing, fake-broker order flow)
```

## Setup

```bash
cd alpaca_bot
pip install -r requirements.txt     # see the note below if this fails on a very new Python
# edit .env: APCA_API_KEY_ID / APCA_API_SECRET_KEY
python -m pytest tests -q           # optional sanity check, no network needed
python -m bot.main                  # or: python bot/main.py
```

Note: `alpaca-trade-api` pins old `websockets`/`msgpack` versions that failed to build on
Python 3.13 when I tried. If `pip install` fails for you, use an older Python or:
`pip install --no-deps alpaca-trade-api==3.2.0 && pip install requests PyYAML deprecation aiohttp websocket-client websockets msgpack python-dotenv pandas numpy`
(the bot only uses the REST client).

The default endpoint is **paper trading**. Pointing `APCA_API_BASE_URL` at the live
endpoint is refused unless `.env` also contains `CONFIRM_LIVE_TRADING=yes`.

## Output files

* `trades.csv` - one row per closed trade: `timestamp` (exit time, UTC), `instrument`,
  `direction`, `entry_price`, `exit_price`, `profit_loss`, `position_size`.
* `daily_pnl.csv` - one row per America/New_York calendar day: starting/ending equity,
  P&L ($ and %), realised P&L from closed trades, trade count. Updated every ~5 min and on
  shutdown; the day rolls over at midnight New York time.
* `bot.log` - everything the bot decides and why. `state.json` - open positions, stop
  levels, cooldowns; lets the bot resume after a restart.

## How the risk rules are implemented

* **ATR sizing**: `qty = (1% of equity) / (stop_atr_mult x ATR(14))`, ATR is Wilder's, computed on
  each strategy's own candles. `stop_atr_mult` is 1 by default and 3 for `daily_trend`, so a hard-stop
  move costs 1% of equity either way (the 3x version just holds a smaller position with a wider stop).
* **Hard stop, always on**: the stop price is set so the loss on the actual position equals
  1% of equity. It is a broker-side stop order (stop-limit for crypto, which Alpaca requires)
  AND re-checked by the bot every 30 s, so it still works if the broker order doesn't fire.
  If the stop order cannot be placed after an entry, the position is closed immediately.
  After shutdown or a crash the broker-side stops stay live.
* **Trailing stops** (2x ATR BTC, 3x ATR GLD/USO) ratchet in the trade's favour only and are
  checked every 30 s against the latest price.
* **Correlation filter**: no new BTC/USD long while SPY and QQQ are both long.
* **Cooldown**: after a hard-stop exit a symbol waits 2 strategy bars before re-entering
  (otherwise mean reversion would re-enter instantly into the same falling price).

## Things you should know

1. **The 1%-per-ATR rule produces huge positions on short-timeframe ATR, so caps apply.**
   Example from live data: SPY ~$770 with a 15-min ATR of roughly $0.7 -> uncapped size is
   ~1,400 shares (~$1.1M, ~10x a $100k account). The bot caps each position at 100% of
   equity (`MAX_POSITION_NOTIONAL_PCT`), total exposure at 200% (`MAX_GROSS_EXPOSURE_PCT`),
   and to available buying power (cash only for crypto). When a cap binds the stop is
   widened so the max loss is *still* exactly 1% of equity - risk is never above 1%, but it
   is not "1% per ATR" in those cases. This will be the normal case for SPY/QQQ/BTC. If you
   want true ATR-scaled sizing, compute ATR on daily candles instead (a small change in
   `bot/main.py`).
2. **Stop vs trailing-stop width.** Because the hard stop (1% of equity) sits about 1 ATR
   away (or closer in % terms when capped) while the trailing stops are 2-3 ATR, the hard
   stop is what ends most losing trades; the trailing stop only matters once a trade has
   moved well in your favour. That is a direct consequence of the spec, not a bug.
3. **No crypto shorts on Alpaca**, and USO is currently hard-to-borrow/not shortable, so:
   BTC/USD "break below the low" only exits a long; a USO cross-down only exits a long.
   (`allow_short` per instrument in `config.py`.)
4. **Signals are evaluated on completed candles only.** "Price breaks above the high" means
   the candle *closed* above it.
5. **Data**: defaults to the free IEX feed (`ALPACA_DATA_FEED=iex`). IEX volume is a small
   slice of the market, and Alpaca's crypto volume is only its own venue (often a few
   hundredths of a BTC per hour), so the BTC 1.5x-volume filter will be noisy.
6. **Equities trade only during regular hours** (checked with Alpaca's market clock);
   BTC/USD runs 24/7. Positions can be held overnight; their stop orders stay at the broker.
7. 4-hour EMA(200) needs ~400 days of history to settle; the bot fetches that on each check
   and refuses to trade the strategy until it has 260 bars.
8. The unit tests verify logic and order flow against a fake broker, not profitability.
   Run the backtest (below) and then paper-trade before risking money.

## Backtesting

```bash
python -m bot.backtest                  # last 6 months, needs the keys in .env
python -m bot.backtest --months 24      # recommended: the daily strategy trades only a few times a year
python -m bot.backtest --refresh        # ignore the CSV cache in backtest_data/ and re-download
python -m bot.backtest --months 12 --capital 50000 --slippage 0.0005
python -m bot.backtest --synthetic      # fake data; ONLY checks that the engine runs
```

It pulls history for all five instruments at their strategy timeframes (plus warm-up bars
before the window), runs the same strategy / risk code the live bot uses, and prints:
per-instrument trades, win rate, average win/loss, profit factor, max drawdown, Sharpe and
total return; per-strategy rows; the combined portfolio with the correlation filter on (and
off for comparison); exit mix; and a FLAGS section for every row with a negative Sharpe, with
the parameters to look at. Outputs: `backtest_results.png`, `backtest_trades.csv`.

**Simulation assumptions**
* Signals use completed candles and fill at the NEXT candle's open (no look-ahead).
* 0.05% adverse slippage on every fill (entries, exits, stops); $0 commission.
* Stops are checked against each candle's high/low using levels set before that candle. A
  gap through the stop fills at the open (worse than the stop); otherwise at the stop level.
* Same sizing, caps, 1% hard stop, trailing stops, cooldown and correlation filter as live.
  Equities get 2x Reg T buying power; crypto is cash-only.
* Open positions are closed at the final candle (`end_of_test`).
* Sharpe uses daily equity returns, risk-free rate 0 (sqrt(252), or sqrt(365) when BTC is included).

**Not modelled**: partial fills, spread/market impact beyond the flat slippage, borrow fees,
overnight gaps beyond what the candles show, IEX vs SIP volume differences, stop-limit
misses on crypto.

**Reading the results**
* With ATR(15-min) sizing the 200% gross-exposure cap is usually full when SPY and QQQ are
  both long, so the correlation filter often changes nothing: the cap would have refused the
  BTC/USD long anyway. The output says so explicitly when it happens.
* GLD/USO trend following on 4h candles trades only a handful of times in 6 months; rows with
  fewer than 10 trades are marked `*` and are low-confidence.
* A negative Sharpe over one 6-month window is evidence, not proof - don't overfit to it.

## Running it every day (Windows)

A Scheduled Task named `AlpacaBot` runs `run_bot.cmd` (venv python, `python -m bot.main`) at
logon and daily at 06:00 local time; a second copy is ignored while one is running, and a crash
restarts it after 1 minute. Output: `bot.log` (decisions) and `bot_console.log` (crash tracebacks).

```powershell
Start-ScheduledTask AlpacaBot        # start now
Stop-ScheduledTask  AlpacaBot        # stop (open positions keep their broker-side stops)
Get-ScheduledTask   AlpacaBot | Get-ScheduledTaskInfo
Unregister-ScheduledTask AlpacaBot -Confirm:$false   # remove
```

The PC must be awake and logged in. If it sleeps, the bot pauses and only the broker-side hard
stops protect open positions. The daily-trend strategy acts once per daily bar, so quiet days are normal.

"""Monitor page: snapshot maths, heartbeat states, log summary, HTTP safety, bot -> status.json."""
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import config
from bot import monitor
from bot.status import build_snapshot, position_view
from test_daily_trend import env  # noqa: F401  (fixture: bot + fake broker on the shipped config)

NOW = datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc)


def pos(direction="long", entry=100.0, qty=10, stop=96.0):
    return SimpleNamespace(symbol="SPY", direction=direction, qty=qty, entry_price=entry,
                           entry_time="t", strategy="daily_trend", hard_stop_price=stop,
                           trailing_stop=None, stop_order_id="o1")


def test_position_view_long_distances():
    v = position_view(pos(), price=104.0, sma=90.0)
    assert v["unrealized"] == 40.0 and v["unrealized_pct"] == 4.0
    assert v["to_stop_pct"] == pytest.approx(7.692, abs=1e-3)       # (104-96)/104
    assert v["to_sma_pct"] == pytest.approx(13.462, abs=1e-3)       # (104-90)/104


def test_position_view_handles_unknown_price():
    v = position_view(pos(), price=None)
    assert v["unrealized"] is None and v["to_stop_pct"] is None and v["to_sma_pct"] is None


def test_health_states():
    snap = lambda age: build_snapshot(started_at="s", mode="PAPER", market_open=True, equity=1, cash=1,
                                     positions=[], signals={}, now=NOW - timedelta(seconds=age))
    assert monitor.health(snap(20), NOW)["state"] == "ok"
    assert monitor.health(snap(150), NOW)["state"] == "stale"
    assert monitor.health(snap(900), NOW)["state"] == "down"
    assert monitor.health(None, NOW)["state"] == "down"


def test_summarize_log_separates_events_decisions_and_retries():
    lines = [
        "2026-10-08 08:30:00,000 INFO     bot: [daily_trend] SPY bar=2026-10-07T04:00:00+00:00 -> HOLD (x)",
        "2026-10-08 08:30:05,000 INFO     bot.portfolio: FILLED LONG SPY qty=49.0 @ 770.79",
        "2026-10-08 09:00:00,000 WARNING  bot.api_utils: _clock failed (ConnectionError); retry 1/5 in 2.0s",
        "2026-10-09 09:00:00,000 WARNING  bot.api_utils: _clock failed (ConnectionError); retry 3/5 in 8.0s",
        "2026-10-09 09:01:00,000 ERROR    bot: job risk_tick crashed",
    ]
    s = monitor.summarize_log(lines)
    assert len(s["decisions"]) == 1
    assert [e.split()[2] for e in s["events"]] == ["INFO", "ERROR"]          # retry warnings are not "events"
    assert s["warnings_by_day"] == {"2026-10-08": 1, "2026-10-09": 1} and s["deep_retries"] == 1


def test_assemble_with_no_files_reports_down(tmp_path):
    d = monitor.assemble(tmp_path, NOW)
    assert d["health"]["state"] == "down" and d["status"] is None and d["trades"] == []


def test_assemble_computes_total_return(tmp_path):
    (tmp_path / "daily_pnl.csv").write_text(
        "date,starting_equity,ending_equity,pnl,pnl_pct,realized_pnl,trades\n"
        "2026-10-05,100000.0,100500.0,500.0,0.5,0.0,0\n")
    snap = build_snapshot(started_at="s", mode="PAPER", market_open=False, equity=101000.0, cash=5.0,
                          positions=[], signals={}, now=NOW)
    (tmp_path / "status.json").write_text(json.dumps(snap))
    d = monitor.assemble(tmp_path, NOW)
    assert d["total_return_pct"] == 1.0 and d["health"]["state"] == "ok"


def test_http_serves_page_and_api_but_rejects_foreign_host_and_writes():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), monitor.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Trading Bot Monitor" in urllib.request.urlopen(base + "/").read()
        assert "health" in json.loads(urllib.request.urlopen(base + "/api/status").read())
        req = urllib.request.Request(base + "/api/status", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e:                  # no POST handler at all
            urllib.request.urlopen(urllib.request.Request(base + "/api/status", data=b"x", method="POST"))
        assert e.value.code == 501
    finally:
        srv.shutdown()


def test_risk_tick_publishes_status_json(env):  # noqa: F811
    env.bot.run_strategy_cycle("daily_trend")                  # opens SPY, records last_signals
    assert env.bot.risk_tick() is True
    snap = json.loads(config.STATUS_FILE.read_text())
    assert snap["mode"] == "PAPER" and snap["equity"] > 0
    assert [p["symbol"] for p in snap["positions"]] == ["SPY"]
    spy = snap["positions"][0]
    assert spy["to_stop_pct"] > 0 and spy["sma"] is not None and spy["to_sma_pct"] > 0
    assert snap["signals"]["SPY"]["strategy"] == "daily_trend"


def test_startup_seeds_signals_for_display_without_placing_orders(env):  # noqa: F811
    env.bot._seed_signals()
    assert {"SPY", "QQQ", "GLD"} <= set(env.bot.last_signals)      # USO has no bars in the fixture: skipped
    spy = env.bot.last_signals["SPY"]
    assert spy["seeded"] is True and spy["info"].get("sma") is not None
    assert env.api.submitted == []                      # display only: no orders

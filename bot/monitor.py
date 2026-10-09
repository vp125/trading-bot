"""Read-only monitoring page for the trading bot.

    python -m bot.monitor            # http://127.0.0.1:8765

Reads only local files the bot already writes (status.json, state.json, daily_pnl.csv,
trades.csv, bot.log). It never talks to the broker, never holds API keys, has no write
or trade endpoints, and binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

PAGE = Path(__file__).with_name("monitor_page.html")
BASE = ROOT                                 # where the bot writes its files; --dir overrides
OK_AGE_S, STALE_AGE_S = 90, 300            # risk tick is 30s; retries can delay it
EVENT_RE = re.compile(r"\b(ENTRY|FILLED|CLOSED|EXIT|ADOPTED|ERROR|CRITICAL|giving up|cycle failed|"
                      r"hard stop|Shutdown|Stopped|Connected to Alpaca|Bot running)\b")
DECISION_RE = re.compile(r"\] \w+ bar=")
RETRY_RE = re.compile(r"retry (\d)/(\d)")


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_csv(path: Path) -> List[dict]:
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _tail(path: Path, nbytes: int = 400_000) -> List[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - nbytes))
            return fh.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []


def summarize_log(lines: List[str]) -> dict:
    """Events, latest decisions and network-retry counts from bot.log lines."""
    events, decisions, warn_by_day = [], [], {}
    deep_retries = 0
    for line in lines:
        day = line[:10]
        if " WARNING " in line:
            warn_by_day[day] = warn_by_day.get(day, 0) + 1
            m = RETRY_RE.search(line)
            if m and int(m.group(1)) >= 3:
                deep_retries += 1
        if DECISION_RE.search(line):
            decisions.append(line)
        elif EVENT_RE.search(line) and "retry" not in line:
            events.append(line)
    return {"events": events[-30:], "decisions": decisions[-12:],
            "warnings_by_day": warn_by_day, "deep_retries": deep_retries}


def health(snapshot: Optional[dict], now: datetime) -> dict:
    if not snapshot or "updated_at" not in snapshot:
        return {"state": "down", "age_s": None, "detail": "no status.json - bot not running yet?"}
    age = (now - datetime.fromisoformat(snapshot["updated_at"])).total_seconds()
    state = "ok" if age <= OK_AGE_S else "stale" if age <= STALE_AGE_S else "down"
    return {"state": state, "age_s": round(age), "detail": f"last heartbeat {round(age)}s ago"}


def assemble(base: Path = ROOT, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    snap = _read_json(base / "status.json")
    state = _read_json(base / "state.json") or {}
    pnl = _read_csv(base / "daily_pnl.csv")
    start_equity = float(pnl[0]["starting_equity"]) if pnl else None
    day = state.get("day") or {}
    equity = snap.get("equity") if snap else None
    out = {
        "now": now.isoformat(), "health": health(snap, now), "status": snap,
        "start_equity": start_equity,
        "day": {"date": day.get("date"), "start_equity": day.get("start_equity"),
                "realized": day.get("realized"), "trades": day.get("trades")},
        "total_return_pct": round((equity / start_equity - 1) * 100, 3) if equity and start_equity else None,
        "daily_pnl": pnl[-60:], "trades": _read_csv(base / "trades.csv")[-15:],
        "cooldowns": state.get("cooldowns", {}),
        "log": summarize_log(_tail(base / "bot.log")),
    }
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "BotMonitor"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):        # blocks DNS-rebinding reads
            return self._send(403, b"forbidden", "text/plain")
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self._send(200, json.dumps(assemble(BASE), default=str).encode(), "application/json")
        self._send(404, b"not found", "text/plain")

    do_HEAD = do_GET  # noqa: N815

    def log_message(self, *args) -> None:        # keep the console quiet
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only trading bot monitor (localhost only).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default=str(ROOT), help="folder holding status.json, state.json, logs (default: repo root)")
    args = ap.parse_args(argv)
    global BASE
    BASE = Path(args.dir)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Monitor on http://127.0.0.1:{args.port}  (read-only, Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

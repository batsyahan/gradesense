"""
GradeSense grading history (SQLite).

Records are keyed by a random per-browser client id, so each visitor only sees their own history.
Only aggregate counts are shared across visitors. Nothing here is written to logs.
"""
import csv
import io
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .config import resolve

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS gradings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT,
    reference TEXT,
    answer TEXT,
    label TEXT,
    verdict TEXT,
    confidence REAL,
    marks REAL,
    max_marks REAL
);
CREATE INDEX IF NOT EXISTS idx_client ON gradings(client_id, created_at);
"""

FIELDS = ["id", "created_at", "source", "question", "reference", "answer", "label", "verdict",
          "confidence", "marks", "max_marks"]


class History:
    def __init__(self, cfg):
        self.enabled = bool(cfg["history_enabled"])
        self.path = resolve(cfg["db_path"])
        self.error = None
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self._conn() as c:
                    c.executescript(SCHEMA)
            except (OSError, sqlite3.Error) as e:
                self.enabled, self.error = False, f"History disabled: {e}"

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_many(self, client_id, source, records):
        """records: iterable of dicts with question, reference, answer, label, verdict, confidence, marks, max_marks"""
        if not self.enabled or not client_id:
            return 0
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [(client_id, now, source, r["question"], r["reference"], r["answer"], r["label"], r["verdict"],
                 float(r["confidence"]), float(r["marks"]), float(r["max_marks"])) for r in records]
        with _lock, self._conn() as c:
            c.executemany("INSERT INTO gradings (client_id, created_at, source, question, reference, answer, label, "
                          "verdict, confidence, marks, max_marks) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def _where(self, client_id, query):
        sql, args = "client_id = ?", [client_id]
        if query:
            like = f"%{query}%"
            sql += " AND (question LIKE ? OR reference LIKE ? OR answer LIKE ? OR verdict LIKE ?)"
            args += [like, like, like, like]
        return sql, args

    def list(self, client_id, query="", offset=0, limit=25):
        if not self.enabled or not client_id:
            return {"total": 0, "items": []}
        where, args = self._where(client_id, query)
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM gradings WHERE {where}", args).fetchone()[0]
            rows = c.execute(f"SELECT {', '.join(FIELDS)} FROM gradings WHERE {where} "
                             f"ORDER BY id DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return {"total": total, "items": [dict(r) for r in rows]}

    def export_csv(self, client_id):
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(FIELDS)
        if self.enabled and client_id:
            with self._conn() as c:
                for r in c.execute(f"SELECT {', '.join(FIELDS)} FROM gradings WHERE client_id = ? ORDER BY id",
                                   [client_id]):
                    w.writerow([r[f] for f in FIELDS])
        return out.getvalue()

    def delete_client(self, client_id):
        if not self.enabled or not client_id:
            return 0
        with _lock, self._conn() as c:
            return c.execute("DELETE FROM gradings WHERE client_id = ?", [client_id]).rowcount

    def stats(self, client_id=None, days=14):
        """Totals plus a daily series for the last `days` days (all visitors, counts only)."""
        if not self.enabled:
            return {"enabled": False, "total": None, "yours": None, "daily": [], "week_change": None}
        start = (datetime.now(timezone.utc) - timedelta(days=days - 1)).date()
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM gradings").fetchone()[0]
            yours = (c.execute("SELECT COUNT(*) FROM gradings WHERE client_id = ?", [client_id]).fetchone()[0]
                     if client_id else 0)
            rows = c.execute("SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM gradings "
                             "WHERE created_at >= ? GROUP BY d", [start.isoformat()]).fetchall()
        counts = {r["d"]: r["n"] for r in rows}
        daily = [{"date": (start + timedelta(days=i)).isoformat(),
                  "count": counts.get((start + timedelta(days=i)).isoformat(), 0)} for i in range(days)]
        this_week = sum(d["count"] for d in daily[-7:])
        last_week = sum(d["count"] for d in daily[-14:-7]) if days >= 14 else 0
        change = round((this_week - last_week) / last_week, 3) if last_week > 0 else None
        return {"enabled": True, "total": total, "yours": yours, "daily": daily,
                "this_week": this_week, "last_week": last_week, "week_change": change}

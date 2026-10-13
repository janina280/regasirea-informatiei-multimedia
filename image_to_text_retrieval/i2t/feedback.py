"""Feedback de relevanță dat de utilizator rezultatelor (SQLite)."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts REAL NOT NULL,
                       collection TEXT NOT NULL,
                       model TEXT NOT NULL,
                       image_hash TEXT NOT NULL,
                       text TEXT NOT NULL,
                       rank INTEGER NOT NULL,
                       score REAL NOT NULL,
                       relevant INTEGER NOT NULL
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, *, collection: str, model: str, image_hash: str, text: str,
            rank: int, score: float, relevant: bool) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO feedback (ts, collection, model, image_hash, text, rank, score, relevant)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), collection, model, image_hash, text, rank, score, int(relevant)),
            )

    def stats(self) -> dict:
        """Precizia percepută pe poziții și pe modele, calculată din voturile utilizatorilor."""
        with self._lock, self._conn() as c:
            total, rel = c.execute("SELECT COUNT(*), COALESCE(SUM(relevant),0) FROM feedback").fetchone()
            per_model = c.execute(
                "SELECT model, COUNT(*), AVG(relevant) FROM feedback GROUP BY model ORDER BY model"
            ).fetchall()
            per_rank = c.execute(
                "SELECT rank, COUNT(*), AVG(relevant) FROM feedback GROUP BY rank ORDER BY rank LIMIT 10"
            ).fetchall()
        return {
            "total": total,
            "relevant": rel,
            "by_model": [{"model": m, "votes": n, "relevant_rate": r} for m, n, r in per_model],
            "by_rank": [{"rank": k, "votes": n, "relevant_rate": r} for k, n, r in per_rank],
        }

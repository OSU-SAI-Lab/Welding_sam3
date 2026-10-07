"""Who owns which upload and which tracking job.

Authentication alone only keeps strangers out; without this, any signed-in user
who learned another user's upload id could read, re-track or delete their work.
Records are written as the gateway sees successful create calls go past.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from app.config import OWNERSHIP_DB

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = Path(OWNERSHIP_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Starlette serves requests from a thread pool, and
    # every access here is already serialised by _lock.
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ownership (
            kind     TEXT NOT NULL,          -- 'upload' or 'job'
            resource TEXT NOT NULL,
            username TEXT NOT NULL,
            created  REAL NOT NULL DEFAULT (strftime('%s','now')),
            PRIMARY KEY (kind, resource)
        )
        """
    )
    _conn.commit()
    return _conn


def record(kind: str, resource: str, username: str) -> None:
    with _lock:
        conn = _connect()
        # First writer wins: a replayed create must not reassign an existing
        # resource to somebody else.
        conn.execute(
            "INSERT OR IGNORE INTO ownership (kind, resource, username) VALUES (?, ?, ?)",
            (kind, resource, username),
        )
        conn.commit()


def owner_of(kind: str, resource: str) -> str | None:
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT username FROM ownership WHERE kind = ? AND resource = ?",
            (kind, resource),
        ).fetchone()
    return row[0] if row else None


def may_access(kind: str, resource: str, username: str) -> bool:
    """
    True when `username` may use this resource.

    An unrecorded resource is allowed through: the data directory predates this
    gateway, and refusing everything created before it was deployed would strand
    work people still need. Anything created from now on is recorded on the way
    past and is therefore owned.
    """
    owner = owner_of(kind, resource)
    return owner is None or owner == username

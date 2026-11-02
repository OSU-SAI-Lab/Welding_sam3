"""Logging setup and the context that ties a log line to a pipeline and a user.

Without `configure()` only WARNING and above from `app.*` reach the pod log
(Python's last-resort handler), so every progress line the service writes is
lost. `configure()` gives them a timestamped handler and stamps uvicorn's own
access lines with the time as well.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import resource
import sys
from pathlib import Path
from typing import Any

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

# Set by the client (X-Pipeline-Id) and by the auth gateway (X-Auth-User, which
# it overwrites, so the browser cannot claim another user). Used only to label
# log lines and job records, never to authorise anything.
PIPELINE_HEADER = "x-pipeline-id"
USER_HEADER = "x-auth-user"
_SAFE = re.compile(r"[^A-Za-z0-9_.@-]")


def configure() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    # Libraries stay at WARNING so a DEBUG run is not drowned in httpx/PIL noise.
    root.setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(LOG_LEVEL)


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's default config with a timestamp on every line."""
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    for name in ("default", "access"):
        fmt = config["formatters"][name]
        fmt["fmt"] = "%(asctime)s " + fmt["fmt"]
        fmt["datefmt"] = _DATEFMT
    return config


def clean(value: str | None) -> str | None:
    """A header value safe to write into a log line (no newlines, bounded)."""
    if not value:
        return None
    return _SAFE.sub("", value)[:64] or None


def request_context(headers: Any) -> dict[str, str | None]:
    return {
        "pipeline_id": clean(headers.get(PIPELINE_HEADER)),
        "user": clean(headers.get(USER_HEADER)),
    }


def tag(**fields: Any) -> str:
    """`[job=1a2b3c4d pipeline=138 user=alice]`, skipping empty fields."""
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        if key in ("job", "upload"):
            value = str(value)[:8]
        parts.append(f"{key}={value}")
    return "[" + " ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Resource snapshot: the numbers that would have shown the OOM coming.
# ---------------------------------------------------------------------------

_GB = 1024 ** 3


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    # macOS/dev: peak, not current, but better than nothing.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def memory() -> str:
    """`rss=11.2G container=19.8G/32.0G gpu=5.1G peak=7.3G` (fields that exist)."""
    parts = []
    rss = _rss_bytes()
    if rss is not None:
        parts.append(f"rss={rss / _GB:.1f}G")
    # cgroup v2, then v1: what the OOM killer actually compares against the limit.
    used = _read_int("/sys/fs/cgroup/memory.current") or _read_int(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )
    limit = _read_int("/sys/fs/cgroup/memory.max") or _read_int(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    if used is not None:
        parts.append(
            f"container={used / _GB:.1f}G/{limit / _GB:.1f}G" if limit and limit < 1 << 60
            else f"container={used / _GB:.1f}G"
        )
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(
                f"gpu={torch.cuda.memory_allocated() / _GB:.1f}G "
                f"peak={torch.cuda.max_memory_allocated() / _GB:.1f}G"
            )
    except Exception:
        pass
    return " ".join(parts) or "n/a"

"""Runtime configuration for the auth gateway."""

from __future__ import annotations

import os

# The SAM3 video service, which binds localhost inside the pod so that this
# gateway is the only way in.
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:2129").rstrip("/")

# Tapis deployment used to validate the caller's token.
TAPIS_BASE_URL = os.environ.get("TAPIS_BASE_URL", "https://icicleai.tapis.io").rstrip("/")

# Origins allowed to call this gateway with credentials. Browsers reject
# wildcard origins on credentialed requests, so these must be concrete.
ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.environ.get("ALLOWED_ORIGINS", "http://127.0.0.1:5173").split(",")
    if o.strip()
]

# Signs the session cookie. Must be set in production — a generated per-process
# value would invalidate every session on restart and differ between replicas.
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")

SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "sam3_video_session")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(12 * 3600)))

# Cross-site cookies require SameSite=None + Secure, which in turn requires
# HTTPS. Local development over plain http needs Lax + non-secure instead.
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "none").lower()
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "1").strip().lower() in ("1", "true", "yes")

# Ownership records live on the same PVC as the uploads they describe, so they
# survive restarts and stay with the data.
OWNERSHIP_DB = os.environ.get("OWNERSHIP_DB", "/data/auth_gateway/ownership.sqlite3")

HOST = os.environ.get("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))

# How long a validated Tapis token is trusted before re-checking with Tapis.
TOKEN_CACHE_SECONDS = int(os.environ.get("TOKEN_CACHE_SECONDS", "300"))

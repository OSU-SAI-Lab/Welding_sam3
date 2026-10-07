"""Stateless, signed session cookies.

A session is an HMAC-signed `username.expiry.signature` string. Keeping it
stateless means no session store to run and nothing to lose on restart; the
trade-off is that a session cannot be revoked before it expires, which is why
the TTL is hours rather than days.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from app.config import SESSION_SECRET, SESSION_TTL_SECONDS


class SessionError(ValueError):
    pass


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(payload: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())


def issue(username: str, secret: str = SESSION_SECRET, ttl: int = SESSION_TTL_SECONDS) -> str:
    if not secret:
        raise SessionError("SESSION_SECRET is not configured")
    if "." in username:
        # The separator would make the payload ambiguous to parse back.
        raise SessionError("username may not contain '.'")
    payload = f"{username}.{int(time.time()) + ttl}"
    return f"{payload}.{_sign(payload, secret)}"


def verify(cookie: str, secret: str = SESSION_SECRET) -> str:
    """Return the username carried by a valid, unexpired cookie."""
    if not secret:
        raise SessionError("SESSION_SECRET is not configured")
    try:
        username, expiry_raw, signature = cookie.rsplit(".", 2)
    except ValueError as e:
        raise SessionError("malformed session cookie") from e

    expected = _sign(f"{username}.{expiry_raw}", secret)
    # Constant-time comparison: a fast reject on the first wrong byte would leak
    # the signature a byte at a time.
    if not hmac.compare_digest(expected, signature):
        raise SessionError("session signature does not match")

    try:
        expiry = int(expiry_raw)
    except ValueError as e:
        raise SessionError("malformed session expiry") from e
    if expiry < time.time():
        raise SessionError("session has expired")
    return username

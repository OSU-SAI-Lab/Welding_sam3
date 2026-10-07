"""Validating a Tapis token against the Tapis deployment."""

from __future__ import annotations

import time

import httpx

from app.config import TAPIS_BASE_URL, TOKEN_CACHE_SECONDS


class TapisAuthError(Exception):
    pass


# token -> (username, checked_at). Tokens are only exchanged for a session
# cookie, so this is consulted once per sign-in rather than per request; the
# cache just stops a reload storm from hammering Tapis.
_cache: dict[str, tuple[str, float]] = {}


async def username_for_token(token: str) -> str:
    if not token:
        raise TapisAuthError("no token supplied")

    cached = _cache.get(token)
    if cached and time.time() - cached[1] < TOKEN_CACHE_SECONDS:
        return cached[0]

    url = f"{TAPIS_BASE_URL}/v3/oauth2/userinfo"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={"X-Tapis-Token": token})
    except httpx.HTTPError as e:
        raise TapisAuthError(f"could not reach Tapis to validate the token: {e}") from e

    if response.status_code == 401:
        raise TapisAuthError("Tapis rejected the token — sign in again")
    if response.status_code != 200:
        raise TapisAuthError(f"Tapis returned {response.status_code} while validating the token")

    try:
        username = response.json()["result"]["username"]
    except (KeyError, TypeError, ValueError) as e:
        raise TapisAuthError("Tapis response did not contain a username") from e

    _cache[token] = (username, time.time())
    return username

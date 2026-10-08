"""Authenticating gateway in front of the SAM3 video service.

The video service has no concept of users: any caller who can reach it can
upload, track, read and delete anything. This gateway sits in front of it in the
same pod (the service binds localhost, so this is the only way in) and adds the
two things it lacks:

* authentication — the caller proves who they are with a Tapis token once, and
  gets a signed session cookie for subsequent calls;
* ownership — uploads and jobs are recorded against the user who created them,
  so one user cannot reach another's work even knowing its id.

Cookies rather than headers because `EventSource` cannot send custom headers,
and the job-progress stream is server-sent events.

The service itself is unmodified; everything here is additive.
"""

from __future__ import annotations

import logging
import re
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app import ownership, sessions
from app.config import (
    ALLOWED_ORIGINS,
    COOKIE_SAMESITE,
    COOKIE_SECURE,
    HOST,
    PORT,
    SESSION_COOKIE_NAME,
    SESSION_SECRET,
    SESSION_TTL_SECONDS,
    UPSTREAM_URL,
)
from app.tapis import TapisAuthError, token_expiry, username_for_token

logger = logging.getLogger(__name__)
app = FastAPI(title="SAM3 Video Service auth gateway", version="0.1.0")

# Paths served by the gateway itself rather than proxied.
AUTH_SESSION_PATH = "/auth/session"
AUTH_WHOAMI_PATH = "/auth/whoami"

# Reachable without a session: the health probe (kubelet sends no cookies) and
# the sign-in endpoint itself.
OPEN_PATHS = {"/health", AUTH_SESSION_PATH}

UPLOAD_RE = re.compile(r"^/uploads/([^/]+)")
JOB_RE = re.compile(r"^/track-jobs/([^/]+)")
# Hop-by-hop headers must not be forwarded verbatim.
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    if not SESSION_SECRET:
        logger.error("SESSION_SECRET is not set — every authenticated request will be refused")
    # No timeout: tracking SSE streams stay open for the length of a job and a
    # video render can take many minutes.
    _client = httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=None)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Hand-rolled rather than CORSMiddleware because credentialed requests need the
# exact origin echoed back, and the proxied responses carry the upstream's own
# CORS headers which have to be replaced rather than duplicated.

def _cors_headers(origin: str | None) -> dict[str, str]:
    if not origin or origin.rstrip("/") not in ALLOWED_ORIGINS:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }


@app.options("/{path:path}")
async def preflight(path: str, request: Request) -> Response:
    # Preflight carries no cookies by definition, so it is never authenticated.
    origin = request.headers.get("origin")
    headers = _cors_headers(origin)
    if headers:
        headers.update({
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": request.headers.get(
                "access-control-request-headers", "Content-Type, X-Tapis-Token"
            ),
            "Access-Control-Max-Age": "600",
        })
    return Response(status_code=204, headers=headers)


def _json(status: int, body: dict, origin: str | None, **kw) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers=_cors_headers(origin), **kw)


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

@app.post(AUTH_SESSION_PATH)
async def create_session(request: Request) -> Response:
    """Exchange a Tapis token for a session cookie the browser sends on its own."""
    origin = request.headers.get("origin")
    token = request.headers.get("x-tapis-token") or ""
    if not token:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        token = (body or {}).get("token", "")

    try:
        username = await username_for_token(token)
    except TapisAuthError as e:
        return _json(401, {"detail": str(e)}, origin)

    # Never outlive the token this session was minted from: a Tapis token is
    # typically good for a few hours, and a stateless session cannot be revoked
    # early, so the shorter of the two wins.
    ttl = SESSION_TTL_SECONDS
    exp = token_expiry(token)
    if exp is not None:
        ttl = max(0, min(ttl, exp - int(time.time())))
    if ttl <= 0:
        return _json(401, {"detail": "That Tapis token has expired. Sign in again."}, origin)

    try:
        cookie = sessions.issue(username, ttl=ttl)
    except sessions.SessionError as e:
        logger.error("could not issue a session: %s", e)
        return _json(500, {"detail": "session signing is not configured on the server"}, origin)

    response = _json(200, {"username": username, "expires_in": ttl}, origin)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        cookie,
        max_age=ttl,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    return response


@app.delete(AUTH_SESSION_PATH)
async def end_session(request: Request) -> Response:
    response = _json(200, {"detail": "signed out"}, request.headers.get("origin"))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@app.get(AUTH_WHOAMI_PATH)
async def whoami(request: Request) -> Response:
    origin = request.headers.get("origin")
    username = _username_or_none(request)
    if username is None:
        return _json(401, {"detail": "not signed in"}, origin)
    return _json(200, {"username": username}, origin)


def _username_or_none(request: Request) -> str | None:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        return None
    try:
        return sessions.verify(cookie)
    except sessions.SessionError:
        return None


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------

def _authorize(path: str, username: str) -> str | None:
    """Return a refusal reason, or None when the request may proceed."""
    upload = UPLOAD_RE.match(path)
    if upload and not ownership.may_access("upload", upload.group(1), username):
        return "This video belongs to another user."
    job = JOB_RE.match(path)
    if job and not ownership.may_access("job", job.group(1), username):
        return "This tracking job belongs to another user."
    return None


def _record_created_resource(path: str, method: str, username: str, body: bytes) -> None:
    """Note ownership of whatever a successful create call just produced."""
    import json

    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return
    if method == "POST" and path == "/uploads":
        if isinstance(payload, dict) and payload.get("upload_id"):
            ownership.record("upload", str(payload["upload_id"]), username)
    elif method == "POST" and path.endswith("/track"):
        if isinstance(payload, dict) and payload.get("job_id"):
            ownership.record("job", str(payload["job_id"]), username)


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

def _forwarded_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}


def _response_headers(upstream: httpx.Response, origin: str | None) -> dict[str, str]:
    headers = {
        k: v for k, v in upstream.headers.items()
        # The upstream sets its own permissive CORS headers; ours replace them,
        # and content-length is wrong once the body is streamed.
        if k.lower() not in HOP_HEADERS and not k.lower().startswith("access-control-")
    }
    headers.pop("content-length", None)
    headers.update(_cors_headers(origin))
    return headers


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    assert _client is not None
    origin = request.headers.get("origin")
    target = "/" + path
    query = request.url.query
    url = f"{target}?{query}" if query else target

    if target in OPEN_PATHS:
        upstream = await _client.request(request.method, url, headers=_forwarded_headers(request))
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream, origin),
        )

    username = _username_or_none(request)
    if username is None:
        return _json(401, {"detail": "Sign in to use the video service."}, origin)

    refusal = _authorize(target, username)
    if refusal:
        return _json(403, {"detail": refusal}, origin)

    # Create calls have small JSON responses that must be read to record who owns
    # the new resource; everything else is streamed so that multi-gigabyte
    # uploads, frame images and SSE never sit in memory.
    creates_resource = request.method == "POST" and (target == "/uploads" or target.endswith("/track"))

    if creates_resource:
        upstream = await _client.request(
            request.method, url,
            headers=_forwarded_headers(request),
            content=request.stream(),
        )
        if upstream.status_code < 300:
            _record_created_resource(target, request.method, username, upstream.content)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream, origin),
        )

    upstream_request = _client.build_request(
        request.method, url,
        headers=_forwarded_headers(request),
        content=request.stream(),
    )
    upstream = await _client.send(upstream_request, stream=True)

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream, origin),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)

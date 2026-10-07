# Auth gateway

The SAM3 video service has no concept of users: any caller that can reach it can
upload, track, read and delete anything. That is fine behind an SSH tunnel,
where SSH is the authentication, but the NRP deployment publishes it on a public
HTTPS hostname — where it meant anyone could fill the disk, occupy the GPU (jobs
run strictly one at a time, so one stranger can starve everyone else), or read
and delete another user's work.

This gateway runs as a sidecar in the same pod and is the only way in. It adds:

* **authentication** — the caller proves who they are once with a Tapis token
  and gets a signed session cookie;
* **ownership** — uploads and jobs are recorded against the user who created
  them, so one user cannot reach another's work even knowing its id.

The SAM3 service itself is unmodified.

## Why cookies rather than a bearer header

Job progress is delivered over server-sent events, and `EventSource` cannot send
custom headers — a token in `Authorization` or `X-Tapis-Token` would leave the
progress stream unauthenticated or broken. A cookie is sent automatically by
both `fetch` (with `credentials: "include"`) and `EventSource` (with
`withCredentials: true`).

Because the frontend and the service are on different origins, the cookie is
`SameSite=None; Secure`, which requires HTTPS, and CORS must echo the exact
calling origin — a browser refuses a wildcard origin on credentialed requests.
Hence `ALLOWED_ORIGINS` must list concrete origins.

## Endpoints it owns

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/auth/session` | Exchange a Tapis token (`X-Tapis-Token` header) for a session cookie |
| DELETE | `/auth/session` | Sign out |
| GET | `/auth/whoami` | The signed-in username |

`GET /health` passes straight through unauthenticated, because kubelet probes
send no cookies. Everything else is proxied only for a signed-in caller who owns
the resource in the path.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `UPSTREAM_URL` | `http://127.0.0.1:2129` | The SAM3 service inside the pod |
| `TAPIS_BASE_URL` | `https://icicleai.tapis.io` | Used to validate tokens |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:5173` | Comma-separated exact origins allowed to call with credentials |
| `SESSION_SECRET` | — | **Required.** Signs session cookies; unset refuses every request |
| `SESSION_TTL_SECONDS` | `43200` | Session lifetime (12 h) |
| `COOKIE_SAMESITE` / `COOKIE_SECURE` | `none` / `1` | Must be `lax` / `0` for local http development |
| `OWNERSHIP_DB` | `/data/auth_gateway/ownership.sqlite3` | Ownership records, on the shared PVC |

## Running locally

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
UPSTREAM_URL=http://127.0.0.1:2129 \
ALLOWED_ORIGINS=http://localhost:5173 \
SESSION_SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))") \
COOKIE_SECURE=0 COOKIE_SAMESITE=lax OWNERSHIP_DB=/tmp/ownership.sqlite3 \
./.venv/bin/uvicorn app.main:app --port 8080
```

Point the frontend at the gateway (`SAM3_VIDEO_URL=http://127.0.0.1:8080`)
rather than at the service.

## Notes and limits

* Sessions are stateless signed cookies, so there is no session store to run and
  nothing is lost on restart — but a session cannot be revoked before it
  expires. The TTL is hours for that reason.
* A resource with no ownership record is allowed through. The data directory
  predates this gateway, and refusing everything created before it was deployed
  would strand work people still need; anything created from now on is recorded
  as it passes and is therefore owned.
* Rotating `SESSION_SECRET` signs everyone out.
* The service still runs one tracking job at a time for the whole deployment.
  This gateway governs who may use it, not how much of it they get.

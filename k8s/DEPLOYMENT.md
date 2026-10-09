# Deploying to NRP (Nautilus)

Both components go into **one pod**: the SAM3 video service and the auth gateway
in front of it. That pairing is not incidental — the gateway reaches the service
over `127.0.0.1`, and the service is never published, so there is no window in
which the GPU is exposed without authentication.

```
                      ┌──────────── pod ─────────────────────┐
  Ingress ──► Service │  auth-gateway :8080                  │
  (HTTPS)     :80 ──► │        │ 127.0.0.1                   │
                      │        ▼                             │
                      │  sam3-video-service :2129            │
                      │  (not published; NetworkPolicy       │
                      │   admits only 8080)                  │
                      └──────────────────────────────────────┘
```

## Before you start

You need: a Nautilus namespace, `kubectl` pointed at it, a registry your
namespace can pull from, a Hugging Face token with access to `facebook/sam3`,
and the frontend's final HTTPS origin.

Decide that origin first. Credentialed requests cannot use a wildcard CORS
origin, so `ALLOWED_ORIGINS` must be the exact origin and getting it wrong
fails every request in the browser with a CORS error rather than a useful
message.

## 1. Build and push both images

The gateway image includes ffmpeg, because the dataset exporter re-extracts
frames from the source video.

```bash
REGISTRY=gitlab-registry.nrp-nautilus.io/<user>/<project>

docker build -t $REGISTRY/sam3-video-service:v0.0.3 \
  -f sam3_video_service/Dockerfile sam3_video_service
docker push $REGISTRY/sam3-video-service:v0.0.3

docker build -t $REGISTRY/sam3-video-auth-gateway:v0.1.0 \
  -f auth_gateway/Dockerfile auth_gateway
docker push $REGISTRY/sam3-video-auth-gateway:v0.1.0
```

Both are `linux/amd64` targets; on an Apple Silicon machine add
`--platform linux/amd64` or the pod will crash-loop on an exec format error.

If the registry is private, create a pull secret and add `imagePullSecrets` to
[deployment.yaml](deployment.yaml):

```bash
kubectl create secret docker-registry regcred \
  --docker-server=$REGISTRY --docker-username=<user> --docker-password=<token>
```

## 2. Fill in the placeholders

| File | Field | Set to |
|------|-------|--------|
| `deployment.yaml` | both `image:` lines | the tags you just pushed |
| `ingress.yaml` | `host` (two places) | your `*.nrp-nautilus.io` hostname |
| `configmap.yaml` | `ALLOWED_ORIGINS` | the frontend's exact HTTPS origin |
| `configmap.yaml` | `TAPIS_BASE_URL` | your Tapis deployment |
| `kustomization.yaml` | `namespace` | your NRP namespace |

Leave `CORS_ALLOW_ORIGINS` as `http://127.0.0.1:8080`. The video service's only
client is the gateway beside it in the pod; it never sees a browser.

## 3. Secrets

```bash
cp secret.example.yaml secret.yaml
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # SESSION_SECRET
# edit secret.yaml: HF_TOKEN and SESSION_SECRET
kubectl apply -f secret.yaml
```

`secret.yaml` is git-ignored. An unset `SESSION_SECRET` makes the gateway refuse
every authenticated request by design, rather than falling back to something
insecure — so if everything returns 401 after deploying, check this first.

## 4. Deploy

```bash
kubectl kustomize .                      # inspect what will be applied
kubectl apply -k .
kubectl rollout status deployment/sam3-video-service
```

Expect the first rollout to be slow: the pod pulls two images and the service
downloads SAM3 weights into the `hf-cache` volume on its first track request.

## 5. Verify

```bash
HOST=https://<your-host>.nrp-nautilus.io

curl -s $HOST/gateway/health     # the gateway itself
curl -s $HOST/health             # proxied through to the GPU service
curl -si $HOST/uploads/anything | head -1   # must be 401 — auth is on
```

`/health` reports the GPU service's state, including `"mock": false` and
`"device": "cuda"` on a real GPU node. The third call proves the gateway is
actually enforcing: a 200 there would mean traffic is reaching the service
unauthenticated.

Then confirm the service is *not* reachable except through the gateway:

```bash
kubectl run probe --rm -it --image=curlimages/curl --restart=Never -- \
  curl -s -m 5 http://sam3-video-service.<namespace>.svc.cluster.local:2129/health
```

That should fail. If it succeeds, the NetworkPolicy is not being enforced —
check that your namespace's CNI supports it before treating the deployment as
secured.

## 6. Point the frontend at it

```bash
SAM3_VIDEO_URL=https://<your-host>.nrp-nautilus.io
SAM3_VIDEO_AUTH=gateway
```

`SAM3_VIDEO_AUTH=gateway` is what turns on cookie credentials and the sign-in
step. Without it the client sends no cookies and every request is refused; with
it set against a deployment that has *no* gateway, the service's wildcard CORS
origin collides with credentials and every request fails in the browser. The
flag must match the deployment.

The frontend must be served over HTTPS. The session cookie is
`SameSite=None; Secure`, which browsers only accept over TLS — necessary because
the frontend and the service are different origins.

## Operating notes

**Probes.** Readiness proxies to the video service, so the pod only takes
traffic once the GPU service answers; liveness checks `/gateway/health`, which
does not touch the upstream. That split matters: a single probe on `/health`
would restart the gateway every time the service was slow to start. The SAM3
container deliberately has no probes of its own — port 2129 is blocked by the
NetworkPolicy, and a kubelet probe against it can be dropped by the CNI, leaving
the pod permanently un-ready.

**One replica, and it must stay that way.** Tracking jobs and ownership live
partly in process memory, the GPU is single, and `strategy: Recreate` avoids two
pods contending for the same PVC. Scaling up will cause jobs to vanish from the
replica that did not run them.

**Disk.** The data volume holds uploads, masks, and staging for dataset exports
(roughly the size of the exported archive, transiently). A 600 MB video export
needs that much free space on top of the upload itself.

**Sessions outlive nothing.** A session is capped at the expiry of the Tapis
token it was minted from, so revoked access cannot linger. Rotating
`SESSION_SECRET` signs everyone out immediately.

**One job at a time.** The service tracks a single job for the whole deployment.
The gateway controls who may use the GPU, not how much of it they get — with
several users, expect queuing.

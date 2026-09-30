# Deploying `sam3_video_service` to NRP (Nautilus)

This deploys only the GPU backend. The Smart Labeler frontend talks to it
over HTTPS via the Ingress below — set `CORS_ALLOW_ORIGINS` in
[configmap.yaml](configmap.yaml) to that frontend's origin once known.

## 1. Build & push the image

```bash
docker build -t <REGISTRY>/sam3-video-service:latest -f sam3_video_service/Dockerfile sam3_video_service
docker push <REGISTRY>/sam3-video-service:latest
```

`<REGISTRY>` is wherever your NRP namespace pulls images from, e.g.
`gitlab-registry.nrp-nautilus.io/<user>/<project>`. If it's a private
registry, also create an `imagePullSecrets` secret and reference it in
[deployment.yaml](deployment.yaml).

## 2. Fill in placeholders

- `deployment.yaml`: `<REGISTRY>/sam3-video-service:latest`
- `ingress.yaml`: `<YOUR-SERVICE>.nrp-nautilus.io` (both `host` entries)
- `kustomization.yaml`: uncomment and set `namespace:` to your NRP namespace

## 3. Secrets

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
# edit k8s/secret.yaml with a real HF_TOKEN (huggingface.co/settings/tokens,
# needs access to facebook/sam3)
kubectl apply -f k8s/secret.yaml
```

`secret.yaml` is git-ignored — never commit it.

## 4. Apply everything else

```bash
kubectl apply -k k8s/
```

This creates:

- `sam3-video-data` / `sam3-hf-cache` PVCs (`rook-cephfs`, adjust size/class
  as needed for your NRP quota)
- a Deployment requesting `nvidia.com/gpu: 1` with `/health` liveness +
  readiness probes
- a ClusterIP Service
- an Ingress on NRP's HAProxy controller (`ingressClassName: haproxy`, TLS
  from NRP's wildcard cert) with a 600s timeout for uploads and tracking jobs

## 5. Verify

```bash
kubectl get pods -w
kubectl logs -f deploy/sam3-video-service
curl https://<YOUR-SERVICE>.nrp-nautilus.io/health
```

`sam3.ready` in the health response stays `false` until the first
`/sessions/{id}/propagate` call finishes loading weights (~1-3 min).

## Notes

- `replicas: 1` / `strategy: Recreate` is intentional — session state lives
  in-process, and only one pod should hold the GPU at a time.
- First real (non-mock) inference call downloads `facebook/sam3` from
  Hugging Face into `/hf-cache` (the `sam3-hf-cache` PVC), so it survives
  pod restarts.
- To point a local Smart Labeler dev instance at this backend, set
  `SAM3_VIDEO_URL=https://<YOUR-SERVICE>.nrp-nautilus.io`.

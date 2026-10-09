#!/usr/bin/env bash
# Build and push both NRP images: the GPU video service and the auth gateway.
#
# Usage:
#   REGISTRY=gitlab-registry.nrp-nautilus.io/<user>/<project> ./scripts/build_push_images.sh [tag]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(date +%Y%m%d-%H%M)}"

if [[ -z "${REGISTRY:-}" ]]; then
  echo "[ERROR] Set REGISTRY, e.g."
  echo "        REGISTRY=gitlab-registry.nrp-nautilus.io/me/sam3 $0"
  exit 1
fi

# Nautilus nodes are amd64. Building on Apple Silicon without this produces an
# arm64 image that crash-loops with an exec format error.
PLATFORM="${PLATFORM:-linux/amd64}"

echo "[INFO] registry : ${REGISTRY}"
echo "[INFO] tag      : ${TAG}"
echo "[INFO] platform : ${PLATFORM}"

build() {
  local name="$1" context="$2" dockerfile="$3"
  echo
  echo "[INFO] building ${name}:${TAG}"
  docker build --platform "${PLATFORM}" \
    -t "${REGISTRY}/${name}:${TAG}" \
    -t "${REGISTRY}/${name}:latest" \
    -f "${dockerfile}" "${context}"
  docker push "${REGISTRY}/${name}:${TAG}"
  docker push "${REGISTRY}/${name}:latest"
}

build sam3-video-service       "${ROOT}/sam3_video_service" "${ROOT}/sam3_video_service/Dockerfile"
build sam3-video-auth-gateway  "${ROOT}/auth_gateway"       "${ROOT}/auth_gateway/Dockerfile"

echo
echo "[INFO] Pushed. Set these in k8s/deployment.yaml:"
echo "         auth-gateway        -> ${REGISTRY}/sam3-video-auth-gateway:${TAG}"
echo "         sam3-video-service  -> ${REGISTRY}/sam3-video-service:${TAG}"

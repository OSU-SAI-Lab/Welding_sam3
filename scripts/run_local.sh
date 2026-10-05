#!/usr/bin/env bash
# Local / cluster: SAM3 video service + labeler UI.
# Mock masks without GPU; real tracking when GPU + HF access are available.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
export DATA_ROOT
export TMPDIR="${DATA_ROOT}/tmp"

# Auto-use real SAM3 when a GPU is available unless explicitly overridden.
if [[ -z "${SAM3_MOCK:-}" ]]; then
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    SAM3_MOCK=0
  else
    SAM3_MOCK=1
  fi
fi
export SAM3_MOCK="${SAM3_MOCK}"
export SAM3_VERSION="${SAM3_VERSION:-sam3}"
export SAM3_BACKEND="${SAM3_BACKEND:-transformers}"
export SAM3_REPO="${SAM3_REPO:-${ROOT}/third_party/sam3}"
export SAM3_VIDEO_URL="${SAM3_VIDEO_URL:-http://127.0.0.1:2129}"

mkdir -p "${DATA_ROOT}" "${TMPDIR}"

# OSC helpers (harmless elsewhere)
if command -v module &>/dev/null; then
  module load python/3.12 2>/dev/null || true
  module load ffmpeg/6.1.1 2>/dev/null || module load ffmpeg 2>/dev/null || true
fi
if ! command -v ffprobe &>/dev/null; then
  echo "[WARN] ffprobe not found — video upload will fail."
  echo "       Install ffmpeg, or on OSC: module load ffmpeg/6.1.1"
fi

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -x "${ROOT}/../.venv-sam3/bin/python" ]]; then
  # Backward-compatible with older welding/ layout
  PY="${ROOT}/../.venv-sam3/bin/python"
  export SAM3_REPO="${SAM3_REPO:-${ROOT}/../sam3}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
else
  echo "[ERROR] No venv found. Run: ./scripts/setup_env.sh"
  exit 1
fi

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] SAM3_MOCK=${SAM3_MOCK} (1=placeholder masks, 0=real SAM3 on GPU)"
echo "[INFO] SAM3_VERSION=${SAM3_VERSION} SAM3_BACKEND=${SAM3_BACKEND}"
echo "[INFO] SAM3_REPO=${SAM3_REPO}"
if [[ "${SAM3_MOCK}" == "0" ]]; then
  echo "[INFO] Real tracking — needs GPU + Hugging Face login (facebook/sam3)"
fi
echo "[INFO] Python: ${PY}"

"${PY}" -m pip install -q -r "${ROOT}/sam3_video_service/requirements.txt"
"${PY}" -m pip install -q -r "${ROOT}/labeler_ui/requirements.txt"

cleanup() {
  kill "${SAM3_PID:-}" "${UI_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "${ROOT}/sam3_video_service"
"${PY}" -m app.main &
SAM3_PID=$!

cd "${ROOT}/labeler_ui"
SAM3_VIDEO_URL="${SAM3_VIDEO_URL}" "${PY}" -m uvicorn app.main:app --host 0.0.0.0 --port 8088 &
UI_PID=$!

echo ""
echo "SAM3 video service: http://127.0.0.1:2129"
echo "Labeler UI:         http://127.0.0.1:8088"
echo "Press Ctrl+C to stop."
wait

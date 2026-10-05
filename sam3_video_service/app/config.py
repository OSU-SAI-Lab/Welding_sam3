"""Runtime configuration for the SAM3 video GPU service."""

from __future__ import annotations

import os
from pathlib import Path

# Shared data root (PVC mount in K8s; local `data/` for dev).
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).expanduser().resolve()

DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# Meta native predictor: sam3 or sam3.1 (facebook/sam3* have no Transformers video integration).
SAM3_VERSION = os.environ.get("SAM3_VERSION", "sam3").strip().lower()
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_DEVICE = os.environ.get("SAM3_DEVICE", "cuda")
# Point-only labeling: disable hotstart heuristics that hide masks on most frames.
SAM3_APPLY_TEMPORAL_DISAMBIGUATION = os.environ.get(
    "SAM3_APPLY_TEMPORAL_DISAMBIGUATION", "0"
).strip().lower() in ("1", "true", "yes")
# meta = build_sam3_predictor from local SAM3 checkout; transformers = HF Sam3TrackerVideoModel.
SAM3_BACKEND = os.environ.get("SAM3_BACKEND", "").strip().lower()
# Path to facebookresearch/sam3 checkout (cloned by scripts/setup_env.sh).
_LABELER_ROOT = Path(__file__).resolve().parents[2]


def _default_sam3_repo() -> Path:
    candidates = [
        _LABELER_ROOT / "third_party" / "sam3",
        _LABELER_ROOT.parent / "sam3",  # older welding/ layout
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


SAM3_REPO = Path(
    os.environ.get("SAM3_REPO", str(_default_sam3_repo()))
).expanduser().resolve()

# Set to 1 to return placeholder masks when SAM3/torch/GPU unavailable (API dev).
SAM3_MOCK = os.environ.get("SAM3_MOCK", "0").strip().lower() in ("1", "true", "yes")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "2129"))

# Comma-separated origins allowed to call this API (e.g. Smart Labeler frontend URL).
# "*" (default) disables credentialed CORS since browsers reject wildcard + credentials.
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]


def sam3_backend() -> str:
    """Return active backend: meta or transformers."""
    if SAM3_BACKEND in ("meta", "transformers"):
        return SAM3_BACKEND
    # SAM 3: HF Sam3TrackerVideoModel is most reliable for point-only chunk labeling.
    if SAM3_VERSION == "sam3":
        return "transformers"
    if SAM3_VERSION == "sam3.1":
        return "meta"
    return "transformers"

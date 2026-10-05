"""Upload/chunk paths on PVC or local DATA_ROOT."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.config import DATA_ROOT


def uploads_root() -> Path:
    root = DATA_ROOT / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def upload_dir(upload_id: str) -> Path:
    path = uploads_root() / upload_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_video_path(upload_id: str) -> Path:
    """Return path to uploaded source video (any common container extension)."""
    d = upload_dir(upload_id)
    meta_file = meta_path(upload_id)
    if meta_file.is_file():
        data = read_json(meta_file)
        name = data.get("source_filename")
        if name:
            path = d / name
            if path.is_file():
                return path
    for candidate in sorted(d.glob("source.*")):
        if candidate.is_file() and candidate.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".m4v",
        }:
            return candidate
    return d / "source.mp4"


def source_video_dest(upload_id: str, original_filename: str) -> Path:
    suffix = Path(original_filename).suffix.lower() or ".mp4"
    return upload_dir(upload_id) / f"source{suffix}"


def export_video_path(upload_id: str) -> Path:
    return upload_dir(upload_id) / "annotated.mp4"


def meta_path(upload_id: str) -> Path:
    return upload_dir(upload_id) / "meta.json"


def chunk_dir(upload_id: str, chunk_index: int) -> Path:
    path = upload_dir(upload_id) / "chunks" / f"chunk_{chunk_index:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_frames_dir(upload_id: str, chunk_index: int) -> Path:
    path = chunk_dir(upload_id, chunk_index) / "frames"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_masks_dir(upload_id: str, chunk_index: int) -> Path:
    path = chunk_dir(upload_id, chunk_index) / "masks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_meta_path(upload_id: str, chunk_index: int) -> Path:
    return chunk_dir(upload_id, chunk_index) / "chunk_meta.json"


def frame_cache_dir(upload_id: str) -> Path:
    path = upload_dir(upload_id) / "frame_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def track_jobs_root() -> Path:
    root = DATA_ROOT / "track_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def track_job_dir(job_id: str) -> Path:
    return track_jobs_root() / job_id


def track_job_masks_dir(job_id: str) -> Path:
    path = track_job_dir(job_id) / "masks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_upload_meta(upload_id: str) -> dict[str, Any]:
    path = meta_path(upload_id)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown upload_id: {upload_id}")
    return read_json(path)


def save_upload_meta(upload_id: str, data: dict[str, Any]) -> None:
    write_json(meta_path(upload_id), data)


def load_chunk_meta(upload_id: str, chunk_index: int) -> dict[str, Any]:
    path = chunk_meta_path(upload_id, chunk_index)
    if not path.is_file():
        raise FileNotFoundError(f"Chunk {chunk_index} not prepared for {upload_id}")
    return read_json(path)


def save_chunk_meta(upload_id: str, chunk_index: int, data: dict[str, Any]) -> None:
    write_json(chunk_meta_path(upload_id, chunk_index), data)


def delete_chunk_frames(upload_id: str, chunk_index: int) -> None:
    frames = chunk_frames_dir(upload_id, chunk_index)
    if frames.is_dir():
        shutil.rmtree(frames)
        frames.mkdir(parents=True, exist_ok=True)

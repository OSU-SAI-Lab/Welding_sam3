"""FastAPI routes for SAM3 video chunk labeling."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app import logs, storage
from app.chunk_planner import plan_chunks
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app import sam3_engine
from app.schemas import (
    FinalizeUploadRequest,
    PointsRequest,
    PrepareChunkRequest,
    PropagateRequest,
    RefineRequest,
    TextPromptRequest,
)
from app.session_manager import session_manager
from app import video_io
from app import video_export
from app import dataset_export

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "sam3": sam3_engine.sam3_status()}


@router.post("/uploads")
async def upload_video(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(400, "filename required")
    upload_id = str(uuid.uuid4())
    dest = storage.source_video_dest(upload_id, file.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        return _register_upload(upload_id, file.filename, dest, logs.request_context(request.headers))
    except OSError as e:
        dest.unlink(missing_ok=True)
        if e.errno == 122:
            raise HTTPException(
                507,
                "Disk quota exceeded — set DATA_ROOT to project storage, e.g. "
                "export DATA_ROOT=/fs/ess/PAS2699/$USER/sam3-video-labeler-data",
            ) from e
        raise HTTPException(500, f"Failed to save upload: {e}") from e


def _register_upload(
    upload_id: str, original_filename: str, dest: Path, context: dict[str, str | None]
) -> dict[str, Any]:
    try:
        info = video_io.probe_video(dest)
    except video_io.VideoIOError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e)) from e

    source_name = dest.name
    meta = {
        "upload_id": upload_id,
        "original_filename": original_filename,
        "source_filename": source_name,
        "frame_count": info["frame_count"],
        "width": info["width"],
        "height": info["height"],
        "duration": info.get("duration"),
        "fps": info.get("fps"),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "status": "uploaded",
        # Who and what the upload is for, so a track job can be traced back to it.
        **context,
    }
    storage.save_upload_meta(upload_id, meta)
    logger.info(
        "%s registered %r: %s frames %sx%s @ %s fps, %.1f MB",
        logs.tag(upload=upload_id, pipeline=context["pipeline_id"], user=context["user"]),
        original_filename, meta["frame_count"], meta["width"], meta["height"], meta["fps"],
        dest.stat().st_size / 1e6,
    )
    return meta


@router.post("/uploads/{upload_id}/finalize")
def finalize_upload(upload_id: str, body: FinalizeUploadRequest, request: Request) -> dict[str, Any]:
    """Register a video already written under DATA_ROOT/uploads/{upload_id}/."""
    dest = storage.source_video_dest(upload_id, body.original_filename)
    if not dest.is_file():
        raise HTTPException(
            404,
            f"Video not found at {dest}. Upload may still be in progress.",
        )
    try:
        return _register_upload(upload_id, body.original_filename, dest, logs.request_context(request.headers))
    except HTTPException:
        raise
    except OSError as e:
        if e.errno == 122:
            raise HTTPException(
                507,
                "Disk quota exceeded — set DATA_ROOT to project storage, e.g. "
                "export DATA_ROOT=/fs/ess/PAS2699/$USER/sam3-video-labeler-data",
            ) from e
        raise HTTPException(500, f"Failed to register upload: {e}") from e


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: str) -> dict[str, Any]:
    try:
        return storage.load_upload_meta(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/uploads/{upload_id}/status")
def upload_status(upload_id: str) -> dict[str, Any]:
    try:
        meta = storage.load_upload_meta(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    plans = plan_chunks(meta["frame_count"], meta.get("chunk_size", DEFAULT_CHUNK_SIZE), meta.get("overlap", DEFAULT_OVERLAP))
    chunks: list[dict[str, Any]] = []
    total_masks = 0
    for plan in plans:
        status = "pending"
        mask_count = 0
        save_start = plan.save_start
        save_end = plan.save_end
        try:
            chunk_meta = storage.load_chunk_meta(upload_id, plan.chunk_index)
            status = chunk_meta.get("status", "prepared")
            save_start = chunk_meta.get("save_start", save_start)
            save_end = chunk_meta.get("save_end", save_end)
            masks_dir = storage.chunk_masks_dir(upload_id, plan.chunk_index)
            mask_count = len(list(masks_dir.glob("*_obj*.json")))
        except FileNotFoundError:
            pass
        total_masks += mask_count
        chunks.append(
            {
                "chunk_index": plan.chunk_index,
                "save_start": save_start,
                "save_end": save_end,
                "status": status,
                "mask_count": mask_count,
            }
        )

    export_ready = storage.export_video_path(upload_id).is_file()
    return {
        "upload_id": upload_id,
        "frame_count": meta["frame_count"],
        "chunks": chunks,
        "total_masks": total_masks,
        "export_ready": export_ready,
        "export_url": f"/uploads/{upload_id}/export/download" if export_ready else None,
    }


@router.post("/uploads/{upload_id}/export")
def start_export(upload_id: str) -> dict[str, Any]:
    try:
        storage.load_upload_meta(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    try:
        out_path = video_export.export_annotated_video(upload_id)
    except video_export.ExportError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return {
        "status": "ready",
        "path": out_path.name,
        "download_url": f"/uploads/{upload_id}/export/download",
    }


@router.get("/uploads/{upload_id}/export/download")
def download_export(upload_id: str):
    from fastapi.responses import FileResponse

    path = storage.export_video_path(upload_id)
    if not path.is_file():
        raise HTTPException(404, "Export not ready — run POST /uploads/{id}/export first")
    meta = storage.load_upload_meta(upload_id)
    name = Path(meta.get("original_filename", "video.mp4")).stem + "_annotated.mp4"
    return FileResponse(path, media_type="video/mp4", filename=name)


@router.post("/uploads/{upload_id}/export/coco")
def export_coco(upload_id: str) -> dict[str, Any]:
    try:
        storage.load_upload_meta(upload_id)
        path = dataset_export.build_coco_json(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except dataset_export.DatasetExportError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return {
        "status": "ready",
        "path": path.name,
        "download_url": f"/uploads/{upload_id}/export/coco/download",
    }


@router.get("/uploads/{upload_id}/export/coco/download")
def download_coco(upload_id: str):
    from fastapi.responses import FileResponse

    path = dataset_export.coco_json_path(upload_id)
    if not path.is_file():
        try:
            path = dataset_export.build_coco_json(upload_id)
        except dataset_export.DatasetExportError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.post("/uploads/{upload_id}/export/yolo")
def export_yolo(upload_id: str) -> dict[str, Any]:
    try:
        storage.load_upload_meta(upload_id)
        path = dataset_export.build_yolo_json(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except dataset_export.DatasetExportError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return {
        "status": "ready",
        "path": path.name,
        "download_url": f"/uploads/{upload_id}/export/yolo/download",
    }


@router.get("/uploads/{upload_id}/export/yolo/download")
def download_yolo(upload_id: str):
    from fastapi.responses import FileResponse

    path = dataset_export.yolo_json_path(upload_id)
    if not path.is_file():
        try:
            path = dataset_export.build_yolo_json(upload_id)
        except dataset_export.DatasetExportError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.post("/uploads/{upload_id}/export/bbox-zip")
def export_bbox_zip(upload_id: str) -> dict[str, Any]:
    try:
        storage.load_upload_meta(upload_id)
        path = dataset_export.build_bbox_overlay_zip(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except dataset_export.DatasetExportError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return {
        "status": "ready",
        "path": path.name,
        "download_url": f"/uploads/{upload_id}/export/bbox-zip/download",
    }


@router.get("/uploads/{upload_id}/export/bbox-zip/download")
def download_bbox_zip(upload_id: str):
    from fastapi.responses import FileResponse

    path = dataset_export.bbox_zip_path(upload_id)
    if not path.is_file():
        try:
            path = dataset_export.build_bbox_overlay_zip(upload_id)
        except dataset_export.DatasetExportError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.get("/uploads/{upload_id}/chunks")
def list_chunks(upload_id: str) -> dict[str, Any]:
    meta = storage.load_upload_meta(upload_id)
    plans = plan_chunks(meta["frame_count"], meta.get("chunk_size", DEFAULT_CHUNK_SIZE), meta.get("overlap", DEFAULT_OVERLAP))
    return {
        "upload_id": upload_id,
        "chunks": [
            {
                "chunk_index": p.chunk_index,
                "process_start": p.process_start,
                "process_end": p.process_end,
                "save_start": p.save_start,
                "save_end": p.save_end,
            }
            for p in plans
        ],
    }


# Preparing a chunk clears its frame directory and refills it with ffmpeg, so
# two requests for the same chunk must not overlap: they delete each other's
# output mid-write. The client can issue a second prepare whenever the first is
# slow — a retry after a gateway timeout, or a user clicking again — so this is
# reached in normal use, not only under contention.
_prepare_locks: dict[tuple[str, int], threading.Lock] = {}
_prepare_locks_guard = threading.Lock()


def _prepare_lock(upload_id: str, chunk_index: int) -> threading.Lock:
    key = (upload_id, chunk_index)
    with _prepare_locks_guard:
        return _prepare_locks.setdefault(key, threading.Lock())


def _already_extracted(upload_id: str, chunk_index: int, frames_dir: Path) -> bool:
    """
    True when a previous prepare finished this chunk and its frames are intact.

    Extraction is the expensive half of prepare — it decodes the video — and
    repeating it for a chunk the client has already prepared is pure cost. The
    recorded frame count is what makes this safe: it is written only after a
    successful extraction, so an interrupted run leaves no count to match and
    the chunk is extracted again.
    """
    try:
        meta = storage.load_chunk_meta(upload_id, chunk_index)
    except Exception:
        return False
    expected = meta.get("extracted_frames")
    if meta.get("status") != "prepared" or not expected:
        return False
    return sum(1 for _ in frames_dir.glob("*.jpg")) == expected


@router.post("/uploads/{upload_id}/chunks/prepare")
def prepare_chunk(upload_id: str, body: PrepareChunkRequest) -> dict[str, Any]:
    meta = storage.load_upload_meta(upload_id)
    video_path = storage.source_video_path(upload_id)
    if not video_path.is_file():
        raise HTTPException(404, "source.mp4 missing")

    plans = plan_chunks(meta["frame_count"], meta.get("chunk_size", DEFAULT_CHUNK_SIZE), meta.get("overlap", DEFAULT_OVERLAP))
    if body.chunk_index >= len(plans):
        raise HTTPException(400, f"chunk_index out of range (max {len(plans) - 1})")
    plan = plans[body.chunk_index]

    frames_dir = storage.chunk_frames_dir(upload_id, body.chunk_index)
    with _prepare_lock(upload_id, body.chunk_index):
        if _already_extracted(upload_id, body.chunk_index, frames_dir):
            logger.info("chunk %s/%s already extracted; reusing frames", upload_id, body.chunk_index)
        else:
            video_io.extract_frames(video_path, frames_dir, plan.process_start, plan.process_end)
        extracted = sorted(frames_dir.glob("*.jpg"))
    if not extracted:
        raise HTTPException(400, "ffmpeg extracted no frames — check video format")
    actual_min = int(extracted[0].stem)
    actual_max = int(extracted[-1].stem)
    save_end = min(plan.save_end, actual_max)
    save_start = max(plan.save_start, actual_min)
    process_end = min(plan.process_end, actual_max)

    sample = video_io.pick_sample_frame(save_start, save_end)
    chunk_meta = {
        "chunk_index": plan.chunk_index,
        "process_start": plan.process_start,
        "process_end": process_end,
        "save_start": save_start,
        "save_end": save_end,
        "sample_frame": sample,
        "status": "prepared",
        # Written last, and only on success: _already_extracted trusts this to
        # decide whether the frames on disk are a complete extraction.
        "extracted_frames": len(extracted),
    }
    storage.save_chunk_meta(upload_id, body.chunk_index, chunk_meta)

    session = session_manager.create(
        upload_id=upload_id,
        chunk_index=plan.chunk_index,
        process_start=plan.process_start,
        process_end=process_end,
        save_start=save_start,
        save_end=save_end,
    )

    return {
        "session_id": session.session_id,
        "chunk": chunk_meta,
        "sample_frame_url": f"/uploads/{upload_id}/chunks/{body.chunk_index}/frames/{sample:06d}.jpg",
    }


@router.get("/uploads/{upload_id}/chunks/{chunk_index}/frames/{frame_idx}.jpg")
def get_frame_image(upload_id: str, chunk_index: int, frame_idx: int):
    from fastapi.responses import FileResponse

    path = storage.chunk_frames_dir(upload_id, chunk_index) / f"{frame_idx:06d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "frame not found")
    return FileResponse(path)


@router.post("/uploads/{upload_id}/chunks/{chunk_index}/sample-frame")
def resample_frame(
    upload_id: str,
    chunk_index: int,
    body: dict[str, list[int]] | None = None,
) -> dict[str, int]:
    chunk = storage.load_chunk_meta(upload_id, chunk_index)
    exclude = set((body or {}).get("exclude") or [])
    frame = video_io.pick_sample_frame(
        chunk["save_start"],
        chunk["save_end"],
        exclude=exclude,
    )
    chunk["sample_frame"] = frame
    storage.save_chunk_meta(upload_id, chunk_index, chunk)
    return {"sample_frame": frame}


@router.post("/sessions/{session_id}/prompt/points")
def add_points(session_id: str, body: PointsRequest) -> dict[str, str]:
    try:
        session_manager.add_points(
            session_id,
            body.frame_idx,
            body.obj_id,
            body.points,
            body.replace,
        )
    except KeyError as e:
        raise HTTPException(404, "session not found") from e
    return {"status": "ok"}


@router.post("/sessions/{session_id}/prompt/text")
def add_text(session_id: str, body: TextPromptRequest) -> dict[str, str]:
    try:
        session_manager.set_text(session_id, body.text, body.frame_idx)
    except KeyError as e:
        raise HTTPException(404, "session not found") from e
    return {"status": "ok"}


@router.post("/sessions/{session_id}/refine")
def refine(session_id: str, body: RefineRequest) -> dict[str, Any]:
    try:
        session = session_manager.get(session_id)
        result = sam3_engine.refine_frame(session, body.frame_idx, body.obj_id)
    except KeyError as e:
        raise HTTPException(404, "session not found") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("refine failed session=%s frame=%s", session_id, body.frame_idx)
        raise HTTPException(500, str(e)) from e
    return result


@router.post("/sessions/{session_id}/propagate")
def start_propagate(session_id: str, body: PropagateRequest) -> dict[str, str]:
    try:
        session_manager.get(session_id)
    except KeyError as e:
        raise HTTPException(404, "session not found") from e
    job_id = str(uuid.uuid4())
    _propagate_jobs[job_id] = {"session_id": session_id, "direction": body.direction}
    return {"job_id": job_id}


_propagate_jobs: dict[str, dict[str, str]] = {}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = _propagate_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    session_id = job["session_id"]
    direction = job.get("direction", "both")

    async def generate():
        try:
            session = session_manager.get(session_id)
            total = session.save_end - session.save_start + 1
            count = 0
            yield _sse({"type": "start", "total": total})

            results_q: queue.Queue = queue.Queue()

            def _run_propagate() -> None:
                try:
                    for result in sam3_engine.propagate_chunk(session, direction=direction):
                        results_q.put(("frame", result))
                    results_q.put(("done", None))
                except Exception as exc:
                    results_q.put(("error", exc))

            worker = threading.Thread(target=_run_propagate, daemon=True)
            worker.start()

            while True:
                try:
                    kind, payload = results_q.get(timeout=10)
                except queue.Empty:
                    yield _sse({"type": "heartbeat", "message": "propagating…"})
                    await asyncio.sleep(0)
                    continue
                if kind == "error":
                    raise payload
                if kind == "done":
                    break
                result = payload
                count += 1
                yield _sse(
                    {
                        "type": "frame",
                        "frame_idx": result.frame_idx,
                        "total": total,
                        "progress": count,
                        "result": result.model_dump(),
                    }
                )
                await asyncio.sleep(0)

            chunk_meta = storage.load_chunk_meta(session.upload_id, session.chunk_index)
            mask_count = len(list(storage.chunk_masks_dir(session.upload_id, session.chunk_index).glob("*_obj*.json")))
            chunk_meta["status"] = "done" if mask_count >= total else "partial"
            chunk_meta["masks_saved"] = mask_count
            storage.save_chunk_meta(session.upload_id, session.chunk_index, chunk_meta)
            storage.delete_chunk_frames(session.upload_id, session.chunk_index)
            session_manager.delete(session_id)
            yield _sse({"type": "done", "frames_saved": mask_count, "expected": total})
        except Exception as e:
            logger.exception("propagate job failed session=%s", session_id)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

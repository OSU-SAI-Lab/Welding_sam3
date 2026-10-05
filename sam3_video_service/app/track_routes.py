"""Routes for multi-object tracking from keyframe masks (see app.track_jobs)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from app import mask_codec, storage, track_jobs, video_io
from app.schemas import TrackRequest
from app.video_export import ExportError, render_overlay_video

router = APIRouter()

_TERMINAL = ("done", "failed", "cancelled")


def _upload_meta(upload_id: str) -> dict[str, Any]:
    try:
        return storage.load_upload_meta(upload_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


def _job_status(job_id: str) -> dict[str, Any]:
    try:
        return track_jobs.status(job_id)
    except KeyError as e:
        raise HTTPException(404, "track job not found") from e


@router.get("/uploads/{upload_id}/frames/{frame_idx}.jpg")
def get_frame(upload_id: str, frame_idx: int) -> FileResponse:
    """Frame N exactly as the tracker indexes it. Clients should draw keyframe masks on these."""
    meta = _upload_meta(upload_id)
    if not 0 <= frame_idx < int(meta["frame_count"]):
        raise HTTPException(404, f"frame_idx out of range [0, {meta['frame_count'] - 1}]")
    cached = storage.frame_cache_dir(upload_id) / f"{frame_idx:06d}.jpg"
    if not cached.is_file():
        with tempfile.TemporaryDirectory(dir=storage.frame_cache_dir(upload_id)) as tmp:
            try:
                out = video_io.extract_frames(
                    storage.source_video_path(upload_id), Path(tmp), frame_idx, frame_idx
                )
            except video_io.VideoIOError as e:
                raise HTTPException(500, str(e)) from e
            if not out:
                raise HTTPException(404, f"could not decode frame {frame_idx}")
            shutil.move(str(out[0]), cached)
    return FileResponse(cached, media_type="image/jpeg")


@router.post("/uploads/{upload_id}/track")
def start_track(upload_id: str, body: TrackRequest) -> dict[str, Any]:
    _upload_meta(upload_id)
    try:
        job = track_jobs.submit(upload_id, body)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return job.public()


@router.get("/track-jobs/{job_id}")
def get_track_job(job_id: str) -> dict[str, Any]:
    return _job_status(job_id)


@router.delete("/track-jobs/{job_id}")
def cancel_track_job(job_id: str) -> dict[str, Any]:
    try:
        return track_jobs.cancel(job_id)
    except KeyError as e:
        raise HTTPException(404, "track job not found") from e


@router.get("/track-jobs/{job_id}/events")
async def track_job_events(job_id: str) -> StreamingResponse:
    """SSE: a status event whenever progress changes, ending on done/failed/cancelled."""
    _job_status(job_id)

    async def generate():
        last = None
        while True:
            st = track_jobs.status(job_id)
            snapshot = json.dumps(st, sort_keys=True)
            if snapshot != last:
                last = snapshot
                yield f"data: {snapshot}\n\n"
            if st["status"] in _TERMINAL:
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/track-jobs/{job_id}/frames/{frame_idx}")
def get_track_frame(job_id: str, frame_idx: int, include_png: bool = False) -> dict[str, Any]:
    """Every object's mask on one frame as COCO RLE (+ optional PNG base64)."""
    st = _job_status(job_id)
    objects = []
    for obj_id, path in sorted(track_jobs.mask_map(job_id).get(frame_idx, {}).items()):
        mask = track_jobs.load_mask(path)
        entry: dict[str, Any] = {
            "obj_id": obj_id,
            "label": st["objects"].get(str(obj_id)),
            "present": bool(mask.any()),
            "area": int(mask.sum()),
            "bbox": mask_codec.mask_bbox_xywh(mask),
            "rle": mask_codec.encode_rle(mask),
        }
        if include_png:
            entry["png_b64"] = mask_codec.encode_png_b64(mask)
        objects.append(entry)
    return {"job_id": job_id, "frame_idx": frame_idx, "status": st["status"], "objects": objects}


@router.get("/track-jobs/{job_id}/masks/{frame_idx}/{obj_id}.png")
def get_track_mask_png(job_id: str, frame_idx: int, obj_id: int) -> FileResponse:
    _job_status(job_id)
    path = storage.track_job_masks_dir(job_id) / f"{frame_idx:06d}_obj{obj_id}.png"
    if not path.is_file():
        raise HTTPException(404, "no mask for that frame/object")
    return FileResponse(path, media_type="image/png")


@router.get("/track-jobs/{job_id}/coco")
async def get_track_coco(job_id: str) -> dict[str, Any]:
    """COCO instance JSON: one image per tracked frame, RLE segmentation per object."""
    st = _job_status(job_id)
    meta = _upload_meta(st["upload_id"])
    return await run_in_threadpool(_build_coco, job_id, st, meta)


def _build_coco(job_id: str, st: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    label_by_obj = {int(k): v for k, v in st["objects"].items()}
    category_ids = {name: i for i, name in enumerate(sorted(set(label_by_obj.values())), start=1)}
    stem = Path(meta.get("original_filename") or "video").stem
    images, annotations = [], []
    for frame_idx, by_obj in sorted(track_jobs.mask_map(job_id).items()):
        anns = []
        for obj_id, path in sorted(by_obj.items()):
            mask = track_jobs.load_mask(path)
            if not mask.any():
                continue
            anns.append(
                {
                    "id": len(annotations) + len(anns) + 1,
                    "image_id": frame_idx + 1,
                    "category_id": category_ids[label_by_obj.get(obj_id, f"object_{obj_id}")],
                    "track_id": obj_id,
                    "segmentation": mask_codec.encode_rle(mask),
                    "area": int(mask.sum()),
                    "bbox": mask_codec.mask_bbox_xywh(mask),
                    "iscrowd": 0,
                }
            )
        if anns:
            h, w = anns[0]["segmentation"]["size"]
            images.append(
                {
                    "id": frame_idx + 1,
                    "file_name": f"{stem}_{frame_idx:06d}.jpg",
                    "frame_idx": frame_idx,
                    "width": w,
                    "height": h,
                }
            )
            annotations.extend(anns)
    return {
        "info": {
            "description": f"SAM3 track job {job_id}",
            "upload_id": st["upload_id"],
            "fps": meta.get("fps"),
            "frame_count": meta.get("frame_count"),
            "status": st["status"],
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in category_ids.items()],
    }


@router.get("/track-jobs/{job_id}/video")
async def get_track_video(job_id: str) -> FileResponse:
    """Annotated MP4, one color per object. Rendered on first request after the job ends."""
    st = _job_status(job_id)
    if st["status"] != "done":
        raise HTTPException(409, f"job is {st['status']}; video is available when done")
    out_path = storage.track_job_dir(job_id) / "annotated.mp4"
    if not out_path.is_file():
        meta = _upload_meta(st["upload_id"])
        try:
            await run_in_threadpool(
                render_overlay_video,
                storage.source_video_path(st["upload_id"]),
                out_path,
                float(meta.get("fps") or 30.0),
                int(meta["frame_count"]),
                track_jobs.mask_map(job_id),
            )
        except ExportError as e:
            out_path.unlink(missing_ok=True)
            raise HTTPException(500, str(e)) from e
    return FileResponse(out_path, media_type="video/mp4", filename=f"tracked_{job_id[:8]}.mp4")

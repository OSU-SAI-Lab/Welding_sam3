"""Multi-object video tracking from keyframe masks (mask-prompted SAM3 tracker).

A job takes masks for one or more objects on one or more keyframes and tracks
every object through [start_frame, end_frame]:

* Pass 1 walks chunks in order. Each chunk is seeded with the user keyframes that
  fall inside it plus, per object, the last non-empty mask from the previous chunk's
  overlap frames. It propagates forward from each object's first seed and (for
  direction="both") backward to cover frames before an object's first seed.
* Pass 2 (direction="both") walks chunks in reverse to fill earlier chunks for
  objects whose first keyframe is in a later chunk, seeding from the next chunk.

Masks are written to DATA_ROOT/track_jobs/{job_id}/masks/{frame:06d}_obj{id}.png.
Jobs run one at a time on a single worker thread (one GPU).
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app import mask_codec, sam3_engine, storage, video_io
from app.chunk_planner import ChunkPlan, plan_chunks
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, SAM3_DEVICE, SAM3_MOCK, sam3_backend
from app.schemas import TrackRequest

logger = logging.getLogger(__name__)

# obj_id -> abs frame_idx -> bool mask (H, W)
Seeds = dict[int, dict[int, np.ndarray]]


class JobCancelled(Exception):
    pass


@dataclass
class TrackJob:
    job_id: str
    upload_id: str
    start_frame: int
    end_frame: int
    direction: str
    labels: dict[int, str]
    status: str = "queued"  # queued | running | done | failed | cancelled
    phase: str = ""
    chunk: int = 0
    chunks_total: int = 0
    frames_done: int = 0
    frames_total: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_requested: bool = False
    user_seeds: Seeds = field(default_factory=dict, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "upload_id": self.upload_id,
            "status": self.status,
            "phase": self.phase,
            "chunk": self.chunk,
            "chunks_total": self.chunks_total,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "progress": round(self.frames_done / self.frames_total, 4) if self.frames_total else 0.0,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "direction": self.direction,
            "objects": {str(k): v for k, v in sorted(self.labels.items())},
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_jobs: dict[str, TrackJob] = {}
_jobs_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit(upload_id: str, req: TrackRequest) -> TrackJob:
    """Validate + decode prompts synchronously (errors -> ValueError), then enqueue."""
    meta = storage.load_upload_meta(upload_id)
    if not storage.source_video_path(upload_id).is_file():
        raise ValueError("source video missing for upload")
    frame_count = int(meta["frame_count"])
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)

    start = req.start_frame or 0
    end = frame_count - 1 if req.end_frame is None else min(req.end_frame, frame_count - 1)
    if start > end:
        raise ValueError(f"start_frame {start} > end_frame {end}")

    seeds: Seeds = defaultdict(dict)
    labels: dict[int, str] = {}
    for kf in req.keyframes:
        if not start <= kf.frame_idx <= end:
            raise ValueError(f"keyframe {kf.frame_idx} outside tracked range [{start}, {end}]")
        for obj in kf.objects:
            try:
                mask = _decode_mask_input(obj.mask, height, width)
            except mask_codec.MaskDecodeError as e:
                raise ValueError(f"frame {kf.frame_idx} obj {obj.obj_id}: {e}") from e
            if not mask.any():
                raise ValueError(f"frame {kf.frame_idx} obj {obj.obj_id}: mask is empty")
            if kf.frame_idx in seeds[obj.obj_id]:
                raise ValueError(f"duplicate mask for frame {kf.frame_idx} obj {obj.obj_id}")
            seeds[obj.obj_id][kf.frame_idx] = mask
            if obj.label:
                labels.setdefault(obj.obj_id, obj.label)
    for obj_id in seeds:
        labels.setdefault(obj_id, f"object_{obj_id}")

    job = TrackJob(
        job_id=str(uuid.uuid4()),
        upload_id=upload_id,
        start_frame=start,
        end_frame=end,
        direction=req.direction,
        labels=labels,
        user_seeds=dict(seeds),
    )
    job_dir = storage.track_job_dir(job.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    for obj_id, by_frame in seeds.items():
        for frame_idx, mask in by_frame.items():
            _write_mask(job_dir / "prompts", frame_idx, obj_id, mask)
    with _jobs_lock:
        _jobs[job.job_id] = job
    _persist(job)
    _ensure_worker()
    _queue.put(job.job_id)
    return job


def get(job_id: str) -> TrackJob | dict[str, Any]:
    """Live job, or the persisted status of a job from a previous service run."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job
    path = storage.track_job_dir(job_id) / "job.json"
    if not path.is_file():
        raise KeyError(job_id)
    return storage.read_json(path)


def status(job_id: str) -> dict[str, Any]:
    job = get(job_id)
    return job.public() if isinstance(job, TrackJob) else job


def cancel(job_id: str) -> dict[str, Any]:
    job = get(job_id)
    if isinstance(job, TrackJob) and job.status in ("queued", "running"):
        job.cancel_requested = True
    return status(job_id)


def mask_map(job_id: str) -> dict[int, dict[int, Path]]:
    from app.video_export import mask_map_from_dir

    return mask_map_from_dir(storage.track_job_masks_dir(job_id))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    _worker = threading.Thread(target=_worker_loop, name="track-worker", daemon=True)
    _worker.start()


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            continue
        if job.cancel_requested:
            _finish(job, "cancelled")
            continue
        job.status = "running"
        _persist(job)
        try:
            _run(job)
            _finish(job, "done")
        except JobCancelled:
            _finish(job, "cancelled")
        except Exception as e:
            logger.exception("track job %s failed", job.job_id)
            job.error = f"{type(e).__name__}: {e}"
            _finish(job, "failed")
        finally:
            shutil.rmtree(storage.track_job_dir(job.job_id) / "frames", ignore_errors=True)
            _release_gpu_cache()


def _finish(job: TrackJob, status_: str) -> None:
    job.status = status_
    job.phase = ""
    job.finished_at = time.time()
    job.user_seeds = {}
    _persist(job)


def _persist(job: TrackJob) -> None:
    storage.write_json(storage.track_job_dir(job.job_id) / "job.json", job.public())


def _release_gpu_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def _engine() -> str:
    """'mock' or 'tracker'. Never silently falls back to mock on real deployments."""
    if SAM3_MOCK:
        return "mock"
    if sam3_backend() != "transformers":
        raise RuntimeError("mask-prompt tracking requires SAM3_BACKEND=transformers")
    for _ in range(600):  # another request may be loading the model
        if sam3_engine._ensure_sam3():
            return "tracker"
        if not sam3_engine._sam3_loading:
            break
        time.sleep(1)
    raise RuntimeError("SAM3 tracker failed to load — check /health and service logs")


def _run(job: TrackJob) -> None:
    engine = _engine()
    meta = storage.load_upload_meta(job.upload_id)
    video_path = storage.source_video_path(job.upload_id)
    masks_dir = storage.track_job_masks_dir(job.job_id)

    count = job.end_frame - job.start_frame + 1
    plans = [
        ChunkPlan(
            chunk_index=p.chunk_index,
            process_start=p.process_start + job.start_frame,
            process_end=p.process_end + job.start_frame,
            save_start=p.save_start + job.start_frame,
            save_end=p.save_end + job.start_frame,
        )
        for p in plan_chunks(
            count,
            int(meta.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            int(meta.get("overlap", DEFAULT_OVERLAP)),
        )
    ]
    first_key = {obj: min(frames) for obj, frames in job.user_seeds.items()}
    backward = job.direction == "both"
    job.chunks_total = len(plans)
    job.frames_total = count
    job.frames_done = 0

    # Pass 1: chunks in order, carrying each object's latest mask forward.
    for plan in plans:
        _check_cancel(job)
        job.phase = "forward"
        job.chunk = plan.chunk_index
        _persist(job)
        seeds: Seeds = defaultdict(dict)
        for obj, by_frame in job.user_seeds.items():
            for f, m in by_frame.items():
                if plan.process_start <= f <= plan.process_end:
                    seeds[obj][f] = m
        if plan.chunk_index > 0:
            for obj in job.user_seeds:
                carry = _saved_mask_in(masks_dir, obj, plan.process_start, plan.save_start - 1, latest=True)
                if carry is not None:
                    seeds[obj].setdefault(carry[0], carry[1])
        if seeds:
            _track_window(
                job, engine, video_path, plan, dict(seeds), forward=True, backward=backward
            )
        job.frames_done = plan.save_end - job.start_frame + 1
        _persist(job)

    # Pass 2: fill earlier chunks for objects first prompted in a later chunk.
    if backward:
        for plan in reversed(plans[:-1]):
            _check_cancel(job)
            seeds = {}
            for obj, key in first_key.items():
                if key <= plan.save_end:
                    continue
                nxt = _saved_mask_in(masks_dir, obj, plan.save_end + 1, plan.process_end, latest=False)
                if nxt is not None:
                    seeds[obj] = {nxt[0]: nxt[1]}
            if not seeds:
                continue
            job.phase = "backward"
            job.chunk = plan.chunk_index
            _persist(job)
            _track_window(job, engine, video_path, plan, seeds, forward=False, backward=True)


def _track_window(
    job: TrackJob,
    engine: str,
    video_path: Path,
    plan: ChunkPlan,
    seeds: Seeds,
    *,
    forward: bool,
    backward: bool,
) -> None:
    """Track seeded objects over plan.process_*; save masks inside plan.save_*.

    Forward results are kept for frames at/after an object's first seed in this
    window; backward results for frames before it.
    """
    masks_dir = storage.track_job_masks_dir(job.job_id)
    first_seed = {obj: min(fr) for obj, fr in seeds.items()}

    def keep(obj: int, frame: int, reverse: bool) -> bool:
        if not plan.save_start <= frame <= plan.save_end or obj not in first_seed:
            return False
        return frame < first_seed[obj] if reverse else frame >= first_seed[obj]

    need_backward = backward and any(f > plan.save_start for f in first_seed.values())
    if not forward and not need_backward:
        return

    if engine == "mock":
        _mock_window(masks_dir, plan, seeds, keep, forward, need_backward)
        return

    import torch

    frames_dir = storage.track_job_dir(job.job_id) / "frames"
    paths = video_io.extract_frames(video_path, frames_dir, plan.process_start, plan.process_end)
    if not paths:
        raise RuntimeError(f"ffmpeg extracted no frames for {plan.process_start}-{plan.process_end}")
    stems = [int(p.stem) for p in paths]
    if stems[0] != plan.process_start:
        raise RuntimeError(f"frame extraction started at {stems[0]}, expected {plan.process_start}")
    frames = [Image.open(p).convert("RGB") for p in paths]
    win_start = plan.process_start
    win_end = stems[-1]
    frame_w, frame_h = frames[0].size

    processor = sam3_engine._tracker_processor
    model = sam3_engine._tracker_model
    device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=next(model.parameters()).dtype,
    )
    del frames

    # Register every seed as a conditioning frame. One add + forward per frame so
    # all objects prompted on that frame are consumed together.
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for obj, fr in seeds.items():
        for f, m in fr.items():
            if win_start <= f <= win_end:
                by_frame[f][obj] = m
    with torch.inference_mode():
        for f in sorted(by_frame):
            objs = sorted(by_frame[f])
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=f - win_start,
                obj_ids=objs,
                input_masks=[_fit(by_frame[f][o], frame_h, frame_w) for o in objs],
            )
            model(inference_session=session, frame_idx=f - win_start)

        passes: list[tuple[bool, int]] = []
        if forward:
            passes.append((False, min(first_seed.values())))
        if need_backward:
            passes.append((True, max(first_seed.values())))
        for reverse, start in passes:
            for out in model.propagate_in_video_iterator(
                session, start_frame_idx=start - win_start, reverse=reverse
            ):
                _check_cancel(job)
                abs_frame = win_start + int(out.frame_idx)
                obj_ids = list(out.object_ids or session.obj_ids)
                if not any(keep(o, abs_frame, reverse) for o in obj_ids):
                    continue
                masks = processor.post_process_masks(
                    [out.pred_masks.float().cpu()],
                    original_sizes=[[frame_h, frame_w]],
                    binarize=False,
                )[0]
                for i, obj in enumerate(obj_ids):
                    if keep(obj, abs_frame, reverse):
                        mask = masks[i, 0].numpy() > 0.0
                        _write_mask(masks_dir, abs_frame, int(obj), mask)
                if not reverse and plan.save_start <= abs_frame:
                    job.frames_done = max(job.frames_done, abs_frame - job.start_frame + 1)


def _mock_window(
    masks_dir: Path,
    plan: ChunkPlan,
    seeds: Seeds,
    keep: Callable[[int, int, bool], bool],
    forward: bool,
    backward: bool,
) -> None:
    """CPU stand-in: hold the nearest seed mask (previous seed forward, first seed backward)."""
    for obj, fr in seeds.items():
        keys = sorted(fr)
        for f in range(plan.save_start, plan.save_end + 1):
            if forward and keep(obj, f, False):
                src = max(k for k in keys if k <= f)
            elif backward and keep(obj, f, True):
                src = keys[0]
            else:
                continue
            _write_mask(masks_dir, f, obj, fr[src])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cancel(job: TrackJob) -> None:
    if job.cancel_requested:
        raise JobCancelled()


def _decode_mask_input(m, height: int, width: int) -> np.ndarray:
    if m.png_b64 is not None:
        return mask_codec.decode_png_b64(m.png_b64)
    if m.rle is not None:
        return mask_codec.decode_rle(m.rle.model_dump())
    if not (height and width):
        raise mask_codec.MaskDecodeError("video size unknown; send png_b64 or rle instead of polygons")
    return mask_codec.decode_polygons(m.polygons, height, width)


def _fit(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a prompt mask to the decoded frame size if the client sent another size."""
    if mask.shape == (height, width):
        return mask
    logger.warning("resizing prompt mask %s -> %s", mask.shape, (height, width))
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    return np.asarray(img) > 127


def _write_mask(masks_dir: Path, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        masks_dir / f"{frame_idx:06d}_obj{obj_id}.png"
    )


def _saved_mask_in(
    masks_dir: Path, obj_id: int, lo: int, hi: int, *, latest: bool
) -> tuple[int, np.ndarray] | None:
    """Latest (or earliest) non-empty saved mask for obj in [lo, hi]."""
    order = range(hi, lo - 1, -1) if latest else range(lo, hi + 1)
    for f in order:
        path = masks_dir / f"{f:06d}_obj{obj_id}.png"
        if path.is_file():
            mask = load_mask(path)
            if mask.any():
                return f, mask
    return None

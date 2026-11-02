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

from app import logs, mask_codec, sam3_engine, storage, video_io
from app.chunk_planner import ChunkPlan, plan_chunks
from app.config import (
    DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, SAM3_DEVICE, SAM3_MOCK, TRACK_CHUNK_SIZE, sam3_backend,
)
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
    # Who and what the job is for, for tracing it in logs and in job.json.
    pipeline_id: str | None = None
    user: str | None = None
    started_at: float | None = None
    user_seeds: Seeds = field(default_factory=dict, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "upload_id": self.upload_id,
            "pipeline_id": self.pipeline_id,
            "user": self.user,
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
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _log(job: TrackJob, level: int, msg: str, *args: Any, **kw: Any) -> None:
    """Log with the job's context, so `grep job=1a2b3c4d` or `grep pipeline=138` finds it all."""
    tag = logs.tag(job=job.job_id, upload=job.upload_id, pipeline=job.pipeline_id, user=job.user)
    logger.log(level, "%s " + msg, tag, *args, **kw)


def _secs(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


# How often a long propagation reports progress (frames, fps, ETA, memory).
PROGRESS_LOG_SECONDS = 30
# Frames decoded and preprocessed together when building a tracking session.
DECODE_BATCH = 16

_jobs: dict[str, TrackJob] = {}
_jobs_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit(
    upload_id: str, req: TrackRequest, *, pipeline_id: str | None = None, user: str | None = None
) -> TrackJob:
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
        pipeline_id=pipeline_id,
        user=user,
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
    ahead = _queue.qsize()
    _queue.put(job.job_id)
    keyframes = sorted({f for by_frame in seeds.values() for f in by_frame})
    _log(
        job, logging.INFO,
        "queued: %d object(s) %s, keyframes %s, frames %d-%d (%d of %d), %dx%d, direction=%s, %d job(s) ahead",
        len(labels), dict(sorted(labels.items())), keyframes, start, end, end - start + 1,
        frame_count, width, height, req.direction, ahead,
    )
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
    """Stop a job. A queued one ends now; a running one at the worker's next check."""
    job = get(job_id)
    if not isinstance(job, TrackJob):
        # Persisted as active but not live: no worker will ever pick it up, so
        # end it on disk rather than leave the user stuck on "running".
        if job.get("status") in ("queued", "running"):
            _end_orphan(job_id, job, "cancelled", None)
            logger.info(
                "%s cancelled a job left %s by a previous service run",
                logs.tag(job=job_id, pipeline=job.get("pipeline_id"), user=job.get("user")), job["status"],
            )
    else:
        _log(job, logging.INFO, "cancel requested while %s (phase=%s chunk=%d/%d)",
             job.status, job.phase or "-", job.chunk + 1, job.chunks_total)
        # Under the lock so the worker cannot start the job between the two lines.
        with _jobs_lock:
            if job.status in ("queued", "running"):
                job.cancel_requested = True
            # A queued job may sit behind another for a long time; waiting for
            # the worker to reach it would leave the user looking at "queued".
            if job.status == "queued":
                _finish(job, "cancelled")
            else:
                _persist(job)
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
        with _jobs_lock:
            # cancel() has already finished a job cancelled while queued.
            if job.cancel_requested:
                continue
            job.status = "running"
            job.started_at = time.time()
            _persist(job)
        _log(job, logging.INFO, "started after %s in queue; %s",
             _secs(job.started_at - job.created_at), logs.memory())
        try:
            _run(job)
            _finish(job, "done")
        except JobCancelled:
            _finish(job, "cancelled")
        except Exception as e:
            _log(job, logging.ERROR, "failed in phase=%s chunk %d/%d after %d/%d frames",
                 job.phase or "-", job.chunk + 1, job.chunks_total, job.frames_done, job.frames_total,
                 exc_info=True)
            job.error = f"{type(e).__name__}: {e}"
            _finish(job, "failed")
        finally:
            elapsed = time.time() - job.started_at
            _log(job, logging.WARNING if job.status == "failed" else logging.INFO,
                 "%s in %s: %d/%d frames (%.1f fps overall); %s",
                 job.status, _secs(elapsed), job.frames_done, job.frames_total,
                 job.frames_done / elapsed if elapsed > 0 else 0.0, logs.memory())
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


def recover_interrupted() -> int:
    """Fail every job a previous service run left queued or running.

    The queue and worker live in process memory, so a restart (an OOM kill, a
    redeploy) loses them while job.json still says the job is active. Without
    this, such a job reports "running" forever. Safe only because the
    deployment runs a single replica with the Recreate strategy: no other live
    process can own these jobs. Returns the number of jobs failed.
    """
    recovered = 0
    for path in storage.track_jobs_root().glob("*/job.json"):
        try:
            data = storage.read_json(path)
        except (OSError, ValueError):
            logger.warning("unreadable track job state %s; skipping", path)
            continue
        if data.get("status") not in ("queued", "running"):
            continue
        job_id = path.parent.name
        # Logged before _end_orphan, which clears the phase this line reports.
        logger.warning(
            "%s was %s (phase=%s chunk=%s/%s, %s/%s frames) when the service stopped; marking failed",
            logs.tag(job=job_id, upload=data.get("upload_id"), pipeline=data.get("pipeline_id"), user=data.get("user")),
            data["status"], data.get("phase") or "-", (data.get("chunk") or 0) + 1, data.get("chunks_total"),
            data.get("frames_done"), data.get("frames_total"),
        )
        _end_orphan(job_id, data, "failed", "Interrupted: the tracking service restarted. Start tracking again.")
        recovered += 1
    return recovered


def _end_orphan(job_id: str, data: dict[str, Any], status_: str, error: str | None) -> None:
    """Finish a job that exists only on disk, mirroring what _finish does for a live one."""
    data.update(status=status_, phase="", error=error, finished_at=time.time())
    storage.write_json(storage.track_job_dir(job_id) / "job.json", data)
    shutil.rmtree(storage.track_job_dir(job_id) / "frames", ignore_errors=True)


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


def _engine(job: TrackJob) -> str:
    """'mock' or 'tracker'. Never silently falls back to mock on real deployments."""
    if SAM3_MOCK:
        return "mock"
    if sam3_backend() != "transformers":
        raise RuntimeError("mask-prompt tracking requires SAM3_BACKEND=transformers")
    started = time.time()
    for waited in range(600):  # another request may be loading the model
        if sam3_engine._ensure_sam3():
            if waited:
                _log(job, logging.INFO, "SAM3 tracker ready after waiting %s", _secs(time.time() - started))
            return "tracker"
        if not sam3_engine._sam3_loading:
            break
        if waited % 30 == 0:
            _log(job, logging.INFO, "waiting for the SAM3 tracker to finish loading (%s so far)",
                 _secs(time.time() - started))
        _check_cancel(job)
        time.sleep(1)
    cause = sam3_engine._sam3_load_error or "check /health and service logs"
    raise RuntimeError(f"SAM3 tracker failed to load — {cause}")


def _run(job: TrackJob) -> None:
    engine = _engine(job)
    meta = storage.load_upload_meta(job.upload_id)
    video_path = storage.source_video_path(job.upload_id)
    masks_dir = storage.track_job_masks_dir(job.job_id)

    count = job.end_frame - job.start_frame + 1
    chunk_size = min(int(meta.get("chunk_size", DEFAULT_CHUNK_SIZE)), TRACK_CHUNK_SIZE)
    overlap = int(meta.get("overlap", DEFAULT_OVERLAP))
    plans = [
        ChunkPlan(
            chunk_index=p.chunk_index,
            process_start=p.process_start + job.start_frame,
            process_end=p.process_end + job.start_frame,
            save_start=p.save_start + job.start_frame,
            save_end=p.save_end + job.start_frame,
        )
        for p in plan_chunks(count, chunk_size, overlap)
    ]
    first_key = {obj: min(frames) for obj, frames in job.user_seeds.items()}
    backward = job.direction == "both"
    job.chunks_total = len(plans)
    job.frames_total = count
    job.frames_done = 0
    _log(job, logging.INFO, "plan: engine=%s, %d chunk(s) of <=%d frames + %d overlap: %s",
         engine, len(plans), chunk_size, overlap,
         ", ".join(f"{p.process_start}-{p.process_end}" for p in plans))

    # Pass 1: chunks in order, carrying each object's latest mask forward.
    for plan in plans:
        _check_cancel(job)
        job.phase = "forward"
        job.chunk = plan.chunk_index
        _persist(job)
        chunk_started = time.time()
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
            _log(job, logging.INFO, "pass 1 chunk %d/%d frames %d-%d: seeds %s",
                 plan.chunk_index + 1, len(plans), plan.process_start, plan.process_end,
                 {obj: sorted(fr) for obj, fr in sorted(seeds.items())})
            _track_window(
                job, engine, video_path, plan, dict(seeds), forward=True, backward=backward
            )
        else:
            _log(job, logging.INFO, "pass 1 chunk %d/%d frames %d-%d: no seeds yet, skipped",
                 plan.chunk_index + 1, len(plans), plan.process_start, plan.process_end)
        job.frames_done = plan.save_end - job.start_frame + 1
        _persist(job)
        _log(job, logging.INFO, "pass 1 chunk %d/%d done in %s; %d/%d frames (%.0f%%)",
             plan.chunk_index + 1, len(plans), _secs(time.time() - chunk_started),
             job.frames_done, job.frames_total, 100 * job.frames_done / job.frames_total)

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
            chunk_started = time.time()
            _log(job, logging.INFO, "pass 2 chunk %d/%d frames %d-%d: back-filling object(s) %s",
                 plan.chunk_index + 1, len(plans), plan.process_start, plan.process_end, sorted(seeds))
            _track_window(job, engine, video_path, plan, seeds, forward=False, backward=True)
            _log(job, logging.INFO, "pass 2 chunk %d/%d done in %s",
                 plan.chunk_index + 1, len(plans), _secs(time.time() - chunk_started))


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
        _log(job, logging.INFO, "chunk %d/%d (%s): mock tracker, frames %d-%d",
             plan.chunk_index + 1, job.chunks_total, job.phase, plan.process_start, plan.process_end)
        _mock_window(masks_dir, plan, seeds, keep, forward, need_backward)
        return

    import torch

    label = f"chunk {plan.chunk_index + 1}/{job.chunks_total} ({job.phase})"
    step = time.time()
    frames_dir = storage.track_job_dir(job.job_id) / "frames"
    paths = video_io.extract_frames(
        video_path, frames_dir, plan.process_start, plan.process_end,
        should_stop=lambda: job.cancel_requested,
    )
    _check_cancel(job)
    if not paths:
        raise RuntimeError(f"ffmpeg extracted no frames for {plan.process_start}-{plan.process_end}")
    stems = [int(p.stem) for p in paths]
    if stems[0] != plan.process_start:
        raise RuntimeError(f"frame extraction started at {stems[0]}, expected {plan.process_start}")
    _log(job, logging.INFO, "%s: extracted %d frames in %s", label, len(paths), _secs(time.time() - step))
    step = time.time()
    win_start = plan.process_start
    win_end = stems[-1]

    processor = sam3_engine._tracker_processor
    model = sam3_engine._tracker_model
    device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"
    dtype = next(model.parameters()).dtype
    # Passing every frame to init_video_session holds all decoded frames at full
    # resolution plus their float32 tensors at once: ~40 MB per 4K frame, which
    # OOM-killed the pod at 550 frames. Instead, preprocess a few frames at a
    # time and keep only the resized tensor in the model dtype (~6 MB per frame
    # whatever the source resolution).
    session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    session.processed_frames = {}
    for i in range(0, len(paths), DECODE_BATCH):
        _check_cancel(job)
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + DECODE_BATCH]]
        if i == 0:
            frame_w, frame_h = batch[0].size
        processed = processor.video_processor(videos=batch, device="cpu", return_tensors="pt")
        for j, frame in enumerate(processed.pixel_values_videos[0]):
            session.processed_frames[i + j] = frame.to("cpu", dtype=dtype)
        del batch, processed
    session.video_height, session.video_width = frame_h, frame_w
    _log(job, logging.INFO, "%s: decoded %d frames (%dx%d) in %s; session ready on %s; %s",
         label, len(paths), frame_w, frame_h, _secs(time.time() - step), device, logs.memory())
    _check_cancel(job)

    # Register every seed as a conditioning frame. One add + forward per frame so
    # all objects prompted on that frame are consumed together.
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for obj, fr in seeds.items():
        for f, m in fr.items():
            if win_start <= f <= win_end:
                by_frame[f][obj] = m
    step = time.time()
    with torch.inference_mode():
        for f in sorted(by_frame):
            _check_cancel(job)
            objs = sorted(by_frame[f])
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=f - win_start,
                obj_ids=objs,
                input_masks=[_fit(by_frame[f][o], frame_h, frame_w) for o in objs],
            )
            model(inference_session=session, frame_idx=f - win_start)
        _log(job, logging.INFO, "%s: %d seed frame(s) registered in %s",
             label, len(by_frame), _secs(time.time() - step))

        passes: list[tuple[bool, int]] = []
        if forward:
            passes.append((False, min(first_seed.values())))
        if need_backward:
            passes.append((True, max(first_seed.values())))
        for reverse, start in passes:
            direction = "backward" if reverse else "forward"
            span = (start - win_start + 1) if reverse else (win_end - start + 1)
            seen = saved = 0
            empty: dict[int, int] = defaultdict(int)  # obj -> frames where its mask came back empty
            # Where each frame's time goes: the model step (waiting on the iterator),
            # resizing masks to the video size, and encoding + writing the PNGs.
            spent = {"model": 0.0, "resize": 0.0, "write": 0.0}
            step = last_report = mark = time.time()
            for out in model.propagate_in_video_iterator(
                session, start_frame_idx=start - win_start, reverse=reverse
            ):
                now = time.time()
                spent["model"] += now - mark
                _check_cancel(job)
                seen += 1
                if now - last_report >= PROGRESS_LOG_SECONDS:
                    fps = seen / (now - step)
                    _log(job, logging.INFO, "%s: %s propagation %d/%d frames, %.1f fps, eta %s; %s; %s",
                         label, direction, seen, span, fps, _secs((span - seen) / fps if fps else 0),
                         _per_frame(spent, seen), logs.memory())
                    last_report = now
                abs_frame = win_start + int(out.frame_idx)
                obj_ids = list(out.object_ids or session.obj_ids)
                if not any(keep(o, abs_frame, reverse) for o in obj_ids):
                    mark = time.time()
                    continue
                masks = processor.post_process_masks(
                    [out.pred_masks.float().cpu()],
                    original_sizes=[[frame_h, frame_w]],
                    binarize=False,
                )[0]
                resized = time.time()
                spent["resize"] += resized - now
                for i, obj in enumerate(obj_ids):
                    if keep(obj, abs_frame, reverse):
                        mask = masks[i, 0].numpy() > 0.0
                        if not mask.any():
                            empty[int(obj)] += 1
                        _write_mask(masks_dir, abs_frame, int(obj), mask)
                        saved += 1
                if not reverse and plan.save_start <= abs_frame:
                    job.frames_done = max(job.frames_done, abs_frame - job.start_frame + 1)
                mark = time.time()
                spent["write"] += mark - resized
            elapsed = time.time() - step
            # Empty masks are how a lost object shows up: worth seeing without opening the results.
            _log(job, logging.INFO, "%s: %s propagation done, %d frames in %s (%.1f fps), %d masks saved; %s%s",
                 label, direction, seen, _secs(elapsed), seen / elapsed if elapsed > 0 else 0.0, saved,
                 _per_frame(spent, seen),
                 f", empty masks per object {dict(sorted(empty.items()))}" if empty else "")


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


def _per_frame(spent: dict[str, float], frames: int) -> str:
    """'per frame: model 0.41s, resize 0.08s, write 0.12s' from accumulated seconds."""
    n = max(frames, 1)
    return "per frame: " + ", ".join(f"{k} {v / n:.2f}s" for k, v in spent.items())


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

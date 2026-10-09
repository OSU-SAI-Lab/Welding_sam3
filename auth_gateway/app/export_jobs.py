"""Asynchronous export of a labelled frame dataset to HPC storage.

Why this lives in the gateway rather than the browser or the video service:

* The browser is the wrong path. A full frame set is hundreds of megabytes, and
  routing it service -> browser -> Tapis means two hops, the tab must stay open,
  and a refresh loses the lot.
* The video service has no Tapis credentials and no notion of users. The gateway
  already validates the caller's token and mounts the same volume, so it can
  read the data and write as that user.

The frames themselves are not kept on disk — the service removes them after each
tracking job — so they are re-extracted from the source video here with the same
ffmpeg flags the service uses, which is what keeps frame numbering identical.

Everything is shipped as one uncompressed tar. One upload of N bytes beats
thousands of individual Tapis POSTs, and JPEG/PNG payloads do not compress, so
spending CPU on gzip would only slow it down. Unpack on the far side with
`tar xf`.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import DATA_ROOT, OWNERSHIP_DB, UPSTREAM_URL

logger = logging.getLogger(__name__)

# Lives beside the ownership database, on the shared volume, so a job survives a
# gateway restart well enough to be reported on.
_STATE_DIR = Path(OWNERSHIP_DB).parent / "exports"

# Phases, in order. Reported so the UI can say what is happening rather than
# showing one undifferentiated bar for a multi-minute operation.
PHASES = ("extracting", "archiving", "uploading")


class ExportCancelled(Exception):
    pass


@dataclass
class ExportJob:
    export_id: str
    username: str
    upload_id: str
    track_job_id: str | None
    system: str
    dest_dir: str
    filename: str
    status: str = "queued"        # queued | running | done | failed | cancelled
    phase: str = ""
    frames_total: int = 0
    frames_done: int = 0
    bytes_total: int = 0
    bytes_sent: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_requested: bool = False
    # Held in memory only, for the life of the job: it is the caller's Tapis
    # credential and is never written to disk.
    _token: str = field(default="", repr=False)

    def public(self) -> dict[str, Any]:
        # One fraction for the whole job, weighted so the bar does not stall:
        # extraction and archiving are file-count work, the upload is bytes.
        # A finished job is complete by definition — `phase` is cleared when it
        # ends, so without this a successful export reported a half-full bar.
        if self.status == "done":
            fraction = 1.0
        elif self.phase == "uploading" and self.bytes_total:
            fraction = 0.5 + 0.5 * (self.bytes_sent / self.bytes_total)
        elif self.frames_total:
            weight = 0.35 if self.phase == "extracting" else 0.5
            fraction = weight * (self.frames_done / self.frames_total)
        else:
            fraction = 0.0
        return {
            "export_id": self.export_id,
            "status": self.status,
            "phase": self.phase,
            "progress": round(min(fraction, 1.0), 4),
            "frames_total": self.frames_total,
            "frames_done": self.frames_done,
            "bytes_total": self.bytes_total,
            "bytes_sent": self.bytes_sent,
            "upload_id": self.upload_id,
            "track_job_id": self.track_job_id,
            "destination": f"{self.system}:{self.dest_dir.rstrip('/')}/{self.filename}",
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_jobs: dict[str, ExportJob] = {}
_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit(
    *,
    username: str,
    upload_id: str,
    track_job_id: str | None,
    system: str,
    dest_dir: str,
    filename: str,
    token: str,
) -> ExportJob:
    source = DATA_ROOT / "uploads" / upload_id
    if not source.is_dir():
        raise ValueError(f"unknown upload: {upload_id}")
    if track_job_id and not (DATA_ROOT / "track_jobs" / track_job_id).is_dir():
        raise ValueError(f"unknown tracking job: {track_job_id}")

    job = ExportJob(
        export_id=str(uuid.uuid4()),
        username=username,
        upload_id=upload_id,
        track_job_id=track_job_id,
        system=system,
        dest_dir=dest_dir,
        filename=filename or f"{upload_id[:8]}_dataset.tar",
        _token=token,
    )
    with _lock:
        _jobs[job.export_id] = job
    _persist(job)
    _ensure_worker()
    _queue.put(job.export_id)
    return job


def get(export_id: str, username: str) -> dict[str, Any] | None:
    """A job's public state, or None when it is not this user's to see."""
    with _lock:
        job = _jobs.get(export_id)
    if job is not None:
        return job.public() if job.username == username else None
    path = _STATE_DIR / f"{export_id}.json"
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return saved.get("public") if saved.get("username") == username else None


def cancel(export_id: str, username: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(export_id)
    if job is None or job.username != username:
        return None
    if job.status in ("queued", "running"):
        job.cancel_requested = True
    return job.public()


def list_for(username: str, limit: int = 20) -> list[dict[str, Any]]:
    with _lock:
        live = [j.public() for j in _jobs.values() if j.username == username]
    live.sort(key=lambda j: j["created_at"], reverse=True)
    return live[:limit]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    _worker = threading.Thread(target=_worker_loop, name="export-worker", daemon=True)
    _worker.start()


def _worker_loop() -> None:
    while True:
        export_id = _queue.get()
        with _lock:
            job = _jobs.get(export_id)
        if job is None:
            continue
        if job.cancel_requested:
            _finish(job, "cancelled")
            continue
        job.status = "running"
        _persist(job)
        staging = DATA_ROOT / "exports" / job.export_id
        try:
            _run(job, staging)
            _finish(job, "done")
        except ExportCancelled:
            _finish(job, "cancelled")
        except Exception as e:
            logger.exception("export %s failed", job.export_id)
            job.error = f"{type(e).__name__}: {e}"
            _finish(job, "failed")
        finally:
            # Several hundred megabytes of staging must not outlive the job.
            shutil.rmtree(staging, ignore_errors=True)
            job._token = ""


def _finish(job: ExportJob, status: str) -> None:
    job.status = status
    job.phase = ""
    job.finished_at = time.time()
    job._token = ""
    _persist(job)


def _persist(job: ExportJob) -> None:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        (_STATE_DIR / f"{job.export_id}.json").write_text(
            json.dumps({"username": job.username, "public": job.public()})
        )
    except OSError as e:
        logger.warning("could not persist export %s: %s", job.export_id, e)


def _check_cancel(job: ExportJob) -> None:
    if job.cancel_requested:
        raise ExportCancelled()


# ---------------------------------------------------------------------------
# The work
# ---------------------------------------------------------------------------

def _run(job: ExportJob, staging: Path) -> None:
    frames_dir = staging / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Re-extract every frame from the source video ──
    job.phase = "extracting"
    _persist(job)
    source = _source_video(job.upload_id)
    meta = _upload_meta(job.upload_id)
    job.frames_total = int(meta.get("frame_count") or 0)
    _persist(job)
    _extract_frames(job, source, frames_dir)

    _check_cancel(job)

    # ── 2. Gather masks and the COCO document beside them ──
    job.phase = "archiving"
    job.frames_done = 0
    _persist(job)
    if job.track_job_id:
        masks_src = DATA_ROOT / "track_jobs" / job.track_job_id / "masks"
        if masks_src.is_dir():
            shutil.copytree(masks_src, staging / "masks", dirs_exist_ok=True)
        coco = _fetch_coco(job.track_job_id)
        if coco is not None:
            (staging / "annotations.coco.json").write_text(json.dumps(coco, indent=2))

    (staging / "README.txt").write_text(_readme(job, meta))

    # ── 3. One tar, streamed to Tapis ──
    archive = DATA_ROOT / "exports" / f"{job.export_id}.tar"
    _build_tar(job, staging, archive)
    _check_cancel(job)

    job.phase = "uploading"
    job.bytes_total = archive.stat().st_size
    _persist(job)
    _upload(job, archive)
    archive.unlink(missing_ok=True)


def _upload_meta(upload_id: str) -> dict[str, Any]:
    path = DATA_ROOT / "uploads" / upload_id / "meta.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _source_video(upload_id: str) -> Path:
    directory = DATA_ROOT / "uploads" / upload_id
    meta = _upload_meta(upload_id)
    name = meta.get("source_filename")
    if name and (directory / name).is_file():
        return directory / name
    for candidate in sorted(directory.glob("source.*")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no source video for upload {upload_id}")


def _extract_frames(job: ExportJob, source: Path, out_dir: Path) -> None:
    """
    Same flags the video service uses, so frame numbers match the annotations.

    `-vsync 0` is the important one: the default constant-rate sync duplicates
    or drops frames on variable-rate input, which would put every frame out of
    step with its mask.
    """
    cmd = [
        "ffmpeg", "-y", "-i", source.as_posix(),
        "-vsync", "0", "-start_number", "0", "-q:v", "2",
        (out_dir / "%06d.jpg").as_posix(),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while process.poll() is None:
            if job.cancel_requested:
                process.kill()
                raise ExportCancelled()
            # ffmpeg writes files as it goes, so counting them is progress.
            job.frames_done = sum(1 for _ in out_dir.glob("*.jpg"))
            _persist(job)
            time.sleep(1.0)
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg exited {process.returncode} while extracting frames")
    job.frames_done = sum(1 for _ in out_dir.glob("*.jpg"))
    if job.frames_done == 0:
        raise RuntimeError("ffmpeg extracted no frames from the source video")
    # The container's frame count is often a frame or two out; report what exists.
    job.frames_total = job.frames_done
    _persist(job)


def _fetch_coco(track_job_id: str) -> Any | None:
    """Built by the video service, so the exported document matches its own export."""
    try:
        response = httpx.get(f"{UPSTREAM_URL}/track-jobs/{track_job_id}/coco", timeout=600.0)
        if response.status_code == 200:
            return response.json()
        logger.warning("COCO fetch for %s returned %d", track_job_id, response.status_code)
    except httpx.HTTPError as e:
        logger.warning("COCO fetch for %s failed: %s", track_job_id, e)
    return None


def _build_tar(job: ExportJob, staging: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(p for p in staging.rglob("*") if p.is_file())
    job.frames_total = len(members) or 1
    job.frames_done = 0
    # Uncompressed: JPEG and PNG are already compressed, so gzip would burn CPU
    # for almost no saving and slow the whole export down.
    with tarfile.open(archive, "w") as tar:
        for index, member in enumerate(members, start=1):
            _check_cancel(job)
            tar.add(member, arcname=str(member.relative_to(staging)))
            if index % 200 == 0 or index == len(members):
                job.frames_done = index
                _persist(job)


def _upload(job: ExportJob, archive: Path) -> None:
    """
    Streams the archive to Tapis as multipart, without reading it into memory.

    The body is generated by hand because the archive can be hundreds of
    megabytes: httpx would otherwise buffer a whole multipart form in memory.
    """
    from app.config import TAPIS_BASE_URL

    dest = job.dest_dir.strip("/")
    encoded = "/".join(
        part for part in [*[p for p in dest.split("/") if p], job.filename] if part
    )
    url = f"{TAPIS_BASE_URL}/v3/files/ops/{job.system}/{encoded}"

    boundary = f"----gateway{uuid.uuid4().hex}"
    preamble = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{job.filename}"\r\n'
        f"Content-Type: application/x-tar\r\n\r\n"
    ).encode()
    epilogue = f"\r\n--{boundary}--\r\n".encode()
    total = len(preamble) + archive.stat().st_size + len(epilogue)

    def body():
        yield preamble
        with archive.open("rb") as fh:
            while True:
                if job.cancel_requested:
                    raise ExportCancelled()
                chunk = fh.read(4 * 1024 * 1024)
                if not chunk:
                    break
                job.bytes_sent += len(chunk)
                _persist(job)
                yield chunk
        yield epilogue

    response = httpx.post(
        url,
        content=body(),
        headers={
            "X-Tapis-Token": job._token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(total),
        },
        timeout=None,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "Tapis rejected the token part-way through the export. Tokens last a few hours, "
            "so a long export started near expiry can outlive it — sign in again and retry."
        )
    if response.status_code >= 300:
        raise RuntimeError(
            f"Tapis refused the upload ({response.status_code}): {response.text[:300]}"
        )
    job.bytes_sent = job.bytes_total


def _readme(job: ExportJob, meta: dict[str, Any]) -> str:
    return (
        "Labelled frame dataset exported from the SAM3 video labeler.\n\n"
        f"source video     : {meta.get('original_filename', 'unknown')}\n"
        f"frames           : {job.frames_total}\n"
        f"size             : {meta.get('width')}x{meta.get('height')}\n"
        f"fps (average)    : {meta.get('fps')}\n"
        f"upload id        : {job.upload_id}\n"
        f"tracking job     : {job.track_job_id or 'none'}\n"
        f"exported by      : {job.username}\n\n"
        "layout\n"
        "  frames/NNNNNN.jpg            one image per decoded frame, 0-based\n"
        "  masks/NNNNNN_objN.png        binary mask, white = object\n"
        "  annotations.coco.json        COCO instances, track_id = object id\n\n"
        "Frame numbers are decoded-frame indices, not timestamps. They are the\n"
        "same indices the annotations use, so frames/000123.jpg lines up with\n"
        "masks/000123_obj1.png and with image frame_idx 123 in the COCO file.\n"
    )

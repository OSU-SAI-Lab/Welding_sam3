"""Build annotated MP4 from source video + per-frame mask PNGs."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app import storage
from app.chunk_planner import plan_chunks
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    pass


# Per-object overlay colors (RGB), cycled by obj_id.
OBJECT_COLORS = [
    (34, 197, 94),  # green
    (59, 130, 246),  # blue
    (239, 68, 68),  # red
    (234, 179, 8),  # yellow
    (168, 85, 247),  # purple
    (6, 182, 212),  # cyan
    (249, 115, 22),  # orange
    (236, 72, 153),  # pink
]


def object_color(obj_id: int) -> tuple[int, int, int]:
    return OBJECT_COLORS[(obj_id - 1) % len(OBJECT_COLORS)]


# frame_idx -> obj_id -> mask file (.png, or legacy bbox .json)
MaskMap = dict[int, dict[int, Path]]


def _parse_mask_name(path: Path) -> tuple[int, int]:
    frame_part, obj_part = path.stem.split("_obj", 1)
    return int(frame_part), int(obj_part)


def mask_map_from_dir(masks_dir: Path, out: MaskMap | None = None) -> MaskMap:
    out = {} if out is None else out
    if not masks_dir.is_dir():
        return out
    for path in sorted(masks_dir.glob("*_obj*.png")):
        frame_idx, obj_id = _parse_mask_name(path)
        out.setdefault(frame_idx, {})[obj_id] = path
    # Fallback: JSON-only masks from older runs (bbox rectangle).
    for path in sorted(masks_dir.glob("*_obj*.json")):
        frame_idx, obj_id = _parse_mask_name(path)
        out.setdefault(frame_idx, {}).setdefault(obj_id, path)
    return out


def _mask_paths_for_upload(upload_id: str) -> MaskMap:
    meta = storage.load_upload_meta(upload_id)
    plans = plan_chunks(
        meta["frame_count"],
        meta.get("chunk_size", DEFAULT_CHUNK_SIZE),
        meta.get("overlap", DEFAULT_OVERLAP),
    )
    out: MaskMap = {}
    for plan in plans:
        mask_map_from_dir(storage.chunk_masks_dir(upload_id, plan.chunk_index), out)
    return out


def export_annotated_video(upload_id: str) -> Path:
    """Overlay saved masks on source video; returns path to annotated.mp4."""
    meta = storage.load_upload_meta(upload_id)
    video_path = storage.source_video_path(upload_id)
    if not video_path.is_file():
        raise ExportError("source.mp4 missing")

    mask_map = _mask_paths_for_upload(upload_id)
    if not mask_map:
        raise ExportError("No masks found — track at least one chunk first")

    out_path = storage.export_video_path(upload_id)
    render_overlay_video(
        video_path,
        out_path,
        fps=float(meta.get("fps") or 30.0),
        frame_count=int(meta["frame_count"]),
        mask_map=mask_map,
    )

    meta["export_path"] = out_path.name
    meta["export_status"] = "ready"
    storage.save_upload_meta(upload_id, meta)
    return out_path


def render_overlay_video(
    video_path: Path,
    out_path: Path,
    fps: float,
    frame_count: int,
    mask_map: MaskMap,
    alpha: float = 0.45,
) -> Path:
    """Decode every frame, alpha-blend each object's mask in its color, re-encode."""
    with tempfile.TemporaryDirectory(prefix="sam3_export_") as tmp:
        tmp_dir = Path(tmp)
        frames_dir = tmp_dir / "frames"
        annotated_dir = tmp_dir / "annotated"
        frames_dir.mkdir()
        annotated_dir.mkdir()

        pattern_in = (frames_dir / "%06d.jpg").as_posix()
        try:
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path.as_posix(),
                    # One image per decoded frame, matching extract_frames indices.
                    # Default CFR sync duplicates/drops frames on VFR input.
                    "-vsync",
                    "0",
                    "-start_number",
                    "0",
                    "-q:v",
                    "2",
                    pattern_in,
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as e:
            raise ExportError("ffmpeg not found; module load ffmpeg") from e
        except subprocess.CalledProcessError as e:
            raise ExportError(f"ffmpeg frame extract failed: {e.output}") from e

        extracted = sorted(frames_dir.glob("*.jpg"))
        if not extracted:
            raise ExportError("ffmpeg extracted no frames from the source video")

        # ffmpeg may use 1-based names when -start_number is unsupported on older
        # builds. Settle that once, from the first file, rather than per frame: a
        # per-frame fallback silently substitutes frame N+1 wherever N is absent,
        # which shifts every later frame out of step with its masks.
        offset = 0 if (frames_dir / "000000.jpg").is_file() else 1

        # Export the contiguous run from the start. `frame_count` comes from the
        # container's metadata, which routinely overstates the length by a frame
        # or two after a trim or re-encode; trusting it threw the whole export
        # away over a frame that was never there.
        available = 0
        while available < frame_count and (frames_dir / f"{available + offset:06d}.jpg").is_file():
            available += 1

        if available == 0:
            raise ExportError("ffmpeg produced no frame numbered from the start of the video")
        if available < frame_count:
            # A short tail means the metadata overstated the length, which is
            # expected and safe to clamp. A hole with frames beyond it is a real
            # extraction fault, and overlaying past it would misalign the masks.
            highest = int(extracted[-1].stem) - offset
            if highest >= available:
                raise ExportError(
                    f"Missing extracted frame {available}, but frames up to {highest} exist — "
                    f"extraction produced a gap rather than a short video"
                )
            logger.warning(
                "%s decoded %d frames but its metadata claims %d; exporting the %d that exist",
                video_path.name, available, frame_count, available,
            )

        for frame_idx in range(available):
            src = frames_dir / f"{frame_idx + offset:06d}.jpg"
            if not src.is_file():
                raise ExportError(f"Missing extracted frame {frame_idx}")

            base = Image.open(src).convert("RGBA")
            for obj_id, mask_path in sorted(mask_map.get(frame_idx, {}).items()):
                if not mask_path.is_file():
                    continue
                if mask_path.suffix == ".png":
                    mask = Image.open(mask_path).convert("L").resize(base.size, Image.NEAREST)
                    mask_arr = np.array(mask) > 127
                else:
                    mask_arr = _mask_from_json(mask_path, base.size)
                if mask_arr.any():
                    overlay_arr = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
                    overlay_arr[..., :3] = object_color(obj_id)
                    overlay_arr[mask_arr, 3] = int(255 * alpha)
                    base = Image.alpha_composite(base, Image.fromarray(overlay_arr, "RGBA"))

            dst = annotated_dir / f"{frame_idx:06d}.jpg"
            base.convert("RGB").save(dst, quality=92)

        pattern_out = (annotated_dir / "%06d.jpg").as_posix()
        encode_base = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-start_number",
            "0",
            "-i",
            pattern_out,
        ]
        encode_tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path.as_posix()]
        last_err: subprocess.CalledProcessError | None = None
        for codec in ("libx264", "mpeg4", "mjpeg"):
            try:
                subprocess.check_output(
                    encode_base + ["-c:v", codec] + encode_tail,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                break
            except subprocess.CalledProcessError as e:
                last_err = e
        else:
            detail = last_err.output if last_err else "no encoder"
            raise ExportError(f"ffmpeg encode failed: {detail}") from last_err
    return out_path


def _mask_from_json(json_path: Path, size: tuple[int, int]) -> np.ndarray:
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    bbox = data.get("bbox") or []
    if len(bbox) == 4:
        draw.rectangle(bbox, fill=255)
    return np.array(mask) > 127

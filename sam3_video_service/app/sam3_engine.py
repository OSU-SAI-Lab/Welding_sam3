"""SAM3 video tracker. Uses Meta native SAM 3.1 by default; HF transformers for legacy sam3."""

from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from app.config import (
    SAM3_APPLY_TEMPORAL_DISAMBIGUATION,
    SAM3_BACKEND,
    SAM3_DEVICE,
    SAM3_MOCK,
    SAM3_MODEL_ID,
    SAM3_REPO,
    SAM3_VERSION,
    sam3_backend,
)
from app.schemas import FrameMaskResult, PointPrompt
from app.session_manager import ChunkSession
from app import storage

logger = logging.getLogger(__name__)

_tracker_model = None
_tracker_processor = None
_video_model = None
_video_processor = None
_meta_predictor = None
_sam3_ready: bool | None = None
_sam3_loading: bool = False
# Why the last load failed, so callers can report the cause rather than a
# generic "failed to load" that sends everyone digging through pod logs.
_sam3_load_error: str | None = None


def sam3_status() -> dict[str, Any]:
    """Report status without blocking on first model load."""
    import torch

    backend = sam3_backend()
    if SAM3_MOCK:
        return {
            "ready": False,
            "mock": True,
            "loading": False,
            "backend": backend,
            "version": SAM3_VERSION,
            "model_id": SAM3_MODEL_ID,
            "device": SAM3_DEVICE,
        }
    model_label = f"meta/{SAM3_VERSION}" if backend == "meta" else SAM3_MODEL_ID
    return {
        "ready": _sam3_ready is True,
        "mock": False,
        "loading": _sam3_loading,
        "backend": backend,
        "version": SAM3_VERSION,
        "model_id": model_label,
        "device": SAM3_DEVICE if torch.cuda.is_available() else "cpu",
        "tracker_loaded": _tracker_model is not None or _meta_predictor is not None,
        "video_loaded": _video_model is not None,
        "load_error": _sam3_load_error,
    }


def _ensure_sam3(*, load_video: bool = False) -> bool:
    if SAM3_MOCK:
        global _sam3_ready
        _sam3_ready = False
        return False
    if sam3_backend() == "meta":
        return _ensure_meta_sam3()
    return _ensure_transformers_sam3(load_video=load_video)


def _ensure_meta_sam3() -> bool:
    global _sam3_ready, _sam3_loading, _meta_predictor
    if _meta_predictor is not None:
        _sam3_ready = True
        return True
    if _sam3_loading:
        return False
    _sam3_loading = True
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for Meta SAM3")

        repo = SAM3_REPO
        if not (repo / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 repo not found at {repo}")
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from sam3.model_builder import build_sam3_predictor

        logger.info("Loading Meta SAM3 predictor version=%s …", SAM3_VERSION)
        predictor_kwargs: dict[str, Any] = {"async_loading_frames": False}
        if SAM3_VERSION == "sam3":
            predictor_kwargs["apply_temporal_disambiguation"] = SAM3_APPLY_TEMPORAL_DISAMBIGUATION
        _meta_predictor = build_sam3_predictor(version=SAM3_VERSION, **predictor_kwargs)
        logger.info("Meta SAM3 %s ready", SAM3_VERSION)
        _sam3_ready = True
        return True
    except Exception as e:
        logger.warning("Meta SAM3 unavailable (%s); using mock masks", e)
        _sam3_ready = False
        return False
    finally:
        _sam3_loading = False


def _ensure_transformers_sam3(*, load_video: bool = False) -> bool:
    global _sam3_ready, _sam3_loading, _sam3_load_error
    global _tracker_model, _tracker_processor, _video_model, _video_processor
    if _tracker_model is not None and _tracker_processor is not None:
        if not load_video or _video_model is not None:
            _sam3_ready = True
            return True
    if _sam3_loading:
        return False
    _sam3_loading = True
    try:
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Model and processor are built into locals and published together. A
        # processor that failed after its model loaded used to leave the model
        # set and the processor None; the next call then skipped loading and
        # reported ready, and every job died on `None.init_video_session`.
        if _tracker_model is None or _tracker_processor is None:
            logger.info("Loading Sam3TrackerVideoModel on %s …", device)
            model = Sam3TrackerVideoModel.from_pretrained(SAM3_MODEL_ID).to(device, dtype=dtype)
            processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_MODEL_ID)
            model.eval()
            _tracker_model, _tracker_processor = model, processor
            logger.info("Sam3TrackerVideoModel ready")

        if load_video and (_video_model is None or _video_processor is None):
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            logger.info("Loading Sam3VideoModel on %s …", device)
            model = Sam3VideoModel.from_pretrained(SAM3_MODEL_ID).to(device, dtype=dtype)
            processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
            model.eval()
            _video_model, _video_processor = model, processor
            logger.info("Sam3VideoModel ready")

        _sam3_ready = True
        _sam3_load_error = None
        return True
    except Exception as e:
        logger.exception("Transformers SAM3 unavailable; using mock masks")
        _sam3_ready = False
        _sam3_load_error = f"{type(e).__name__}: {e}"
        return False
    finally:
        _sam3_loading = False


def release_session(session: ChunkSession) -> None:
    """Release GPU resources for a chunk session."""
    handle = session.inference_handle
    if not isinstance(handle, dict):
        return
    if handle.get("backend") == "meta" and _meta_predictor is not None:
        sid = handle.get("session_id")
        if sid:
            try:
                _meta_predictor.handle_request({"type": "close_session", "session_id": sid})
            except Exception as e:
                logger.debug("close_session failed: %s", e)


def _chunk_frame_paths(upload_id: str, chunk_index: int) -> list[Path]:
    frames_dir = storage.chunk_frames_dir(upload_id, chunk_index)
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No extracted frames in {frames_dir}")
    return paths


def _load_chunk_frames(upload_id: str, chunk_index: int) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in _chunk_frame_paths(upload_id, chunk_index)]


def _frame_index_from_path(path: Path) -> int:
    return int(path.stem)


def _global_to_local(handle: dict[str, Any], global_frame: int) -> int:
    stems: list[int] = handle["frame_stems"]
    try:
        return stems.index(global_frame)
    except ValueError as e:
        raise ValueError(f"frame {global_frame} not in prepared chunk frames") from e


def _meta_points_from_clicks(clicks: list[PointPrompt], width: int, height: int):
    import torch

    coords = [[p.x / width, p.y / height] for p in clicks]
    labels = [p.label for p in clicks]
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int32),
    )


def _meta_add_prompt(
    handle: dict[str, Any],
    local_idx: int,
    obj_id: int,
    clicks: list[PointPrompt] | None = None,
    text: str | None = None,
    *,
    clear_old_points: bool = True,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "type": "add_prompt",
        "session_id": handle["session_id"],
        "frame_index": local_idx,
        "obj_id": obj_id,
        "clear_old_points": clear_old_points,
    }
    if text:
        req["text"] = text
    if clicks:
        points, point_labels = _meta_points_from_clicks(
            clicks, handle["width"], handle["height"]
        )
        req["points"] = points
        req["point_labels"] = point_labels
    return _meta_predictor.handle_request(req)


def _meta_outputs_to_result(
    frame_idx: int, obj_id: int, outputs: dict[str, Any] | None
) -> tuple[FrameMaskResult, np.ndarray]:
    if not outputs or "out_obj_ids" not in outputs:
        empty = np.zeros((1, 1), dtype=bool)
        return (
            FrameMaskResult(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox=[0, 0, 0, 0],
                confidence=0.0,
                segmentation=[],
            ),
            empty,
        )

    obj_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    masks = outputs["out_binary_masks"]
    probs = outputs.get("out_probs")
    if probs is None:
        probs = outputs.get("out_tracker_probs")
    if probs is None:
        probs = outputs.get("out_sam2_probs")

    mask = None
    confidence = 0.75
    for idx, oid in enumerate(obj_ids.tolist()):
        if int(oid) != obj_id:
            continue
        candidate = np.asarray(masks[idx], dtype=bool)
        if not candidate.any():
            continue
        mask = candidate
        if probs is not None:
            p = np.asarray(probs).reshape(-1)
            if idx < len(p):
                confidence = float(p[idx])
        break

    if mask is None or not mask.any():
        h, w = int(masks.shape[-2]), int(masks.shape[-1]) if masks.size else (1, 1)
        empty = np.zeros((h, w), dtype=bool)
        return (
            FrameMaskResult(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox=[0, 0, 0, 0],
                confidence=0.0,
                segmentation=[],
            ),
            empty,
        )

    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    seg = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]
    return (
        FrameMaskResult(
            frame_idx=frame_idx,
            obj_id=obj_id,
            bbox=[x_min, y_min, x_max, y_max],
            confidence=confidence,
            segmentation=seg,
        ),
        mask,
    )


def init_chunk_session(session: ChunkSession) -> None:
    if session.inference_handle is not None:
        return
    if not _ensure_sam3():
        session.inference_handle = {
            "mock": True,
            "frames": len(_load_chunk_frames(session.upload_id, session.chunk_index)),
        }
        session.engine_kind = "mock"
        return

    if sam3_backend() == "meta":
        paths = _chunk_frame_paths(session.upload_id, session.chunk_index)
        frames_dir = storage.chunk_frames_dir(session.upload_id, session.chunk_index)
        meta = storage.load_upload_meta(session.upload_id)
        w = int(meta.get("width") or 640)
        h = int(meta.get("height") or 480)
        logger.info(
            "Starting Meta SAM3 %s session for chunk %s (%s frames) …",
            SAM3_VERSION,
            session.chunk_index,
            len(paths),
        )
        start_req: dict[str, Any] = {
            "type": "start_session",
            "resource_path": str(frames_dir),
            "offload_video_to_cpu": True,
        }
        # SAM 3.1 multiplex init_state() does not accept offload_state_to_cpu.
        if SAM3_VERSION == "sam3":
            start_req["offload_state_to_cpu"] = True
        resp = _meta_predictor.handle_request(start_req)
        session.inference_handle = {
            "backend": "meta",
            "session_id": resp["session_id"],
            "frame_stems": [_frame_index_from_path(p) for p in paths],
            "width": w,
            "height": h,
            "num_frames": len(paths),
        }
        session.engine_kind = f"meta_{SAM3_VERSION}"
        logger.info("Meta SAM3 session ready")
        return

    import torch

    frames = _load_chunk_frames(session.upload_id, session.chunk_index)
    logger.info(
        "Initializing Transformers SAM3 session for chunk %s (%s frames) …",
        session.chunk_index,
        len(frames),
    )
    device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"
    inference_session = _tracker_processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    session.inference_handle = inference_session
    session.engine_kind = "tracker"
    logger.info("Transformers SAM3 session ready")


def _session_video_size(inference_session) -> tuple[int, int]:
    h = inference_session.video_height
    w = inference_session.video_width
    if hasattr(h, "item"):
        h = int(h.item())
    else:
        h = int(h)
    if hasattr(w, "item"):
        w = int(w.item())
    else:
        w = int(w)
    return h, w


def _mock_mask_array(frame_idx: int, obj_id: int, w: int, h: int) -> np.ndarray:
    cx = (frame_idx * 17 + obj_id * 31) % max(w - 120, 1) + 60
    cy = (frame_idx * 13 + obj_id * 23) % max(h - 120, 1) + 60
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = max(0, cy), min(h, cy + 80)
    x0, x1 = max(0, cx), min(w, cx + 80)
    mask[y0:y1, x0:x1] = True
    return mask


def _mock_bbox(frame_idx: int, obj_id: int, w: int = 640, h: int = 480) -> tuple[FrameMaskResult, np.ndarray]:
    mask = _mock_mask_array(frame_idx, obj_id, w, h)
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return (
        FrameMaskResult(
            frame_idx=frame_idx,
            obj_id=obj_id,
            bbox=[x_min, y_min, x_max, y_max],
            confidence=0.75,
            segmentation=[[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        ),
        mask,
    )


def refine_frame(session: ChunkSession, frame_idx: int, obj_id: int) -> dict[str, Any]:
    init_chunk_session(session)
    if session.engine_kind == "mock":
        meta = storage.load_upload_meta(session.upload_id)
        w = int(meta.get("width") or 640)
        h = int(meta.get("height") or 480)
        result, mask = _mock_bbox(frame_idx, obj_id, w, h)
        return _result_with_preview(result, mask)

    clicks = session.clicks.get(frame_idx, {}).get(obj_id, [])
    if not clicks:
        raise ValueError(f"No points for frame {frame_idx} obj {obj_id}")

    if session.engine_kind and session.engine_kind.startswith("meta_"):
        handle = session.inference_handle
        rel = _global_to_local(handle, frame_idx)
        resp = _meta_add_prompt(handle, rel, obj_id, clicks, clear_old_points=True)
        result, mask = _meta_outputs_to_result(frame_idx, obj_id, resp.get("outputs"))
        return _result_with_preview(result, mask)

    inference_session = session.inference_handle
    rel_frame = frame_idx - session.process_start
    num_frames = int(getattr(inference_session, "num_frames", 0) or 0)
    if rel_frame < 0 or (num_frames and rel_frame >= num_frames):
        raise ValueError(
            f"frame {frame_idx} is outside prepared chunk "
            f"(process {session.process_start}–{session.process_end}, {num_frames} frames loaded)"
        )

    pts = [[[[p.x, p.y] for p in clicks]]]
    labels = [[[p.label for p in clicks]]]

    _tracker_processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=rel_frame,
        obj_ids=obj_id,
        input_points=pts,
        input_labels=labels,
    )
    outputs = _tracker_model(
        inference_session=inference_session,
        frame_idx=rel_frame,
    )
    vh, vw = _session_video_size(inference_session)
    masks = _tracker_processor.post_process_masks(
        [outputs.pred_masks.cpu()],
        original_sizes=[[vh, vw]],
        binarize=False,
    )[0]
    result, mask = _mask_tensor_to_result(frame_idx, obj_id, masks, outputs, obj_index=0)
    return _result_with_preview(result, mask)


def _output_confidence(outputs, obj_index: int = 0) -> float:
    import torch

    if getattr(outputs, "iou_scores", None) is not None:
        scores = outputs.iou_scores.detach().cpu()
        while scores.ndim > 1:
            scores = scores[0]
        if scores.numel():
            return float(scores.max())

    if getattr(outputs, "object_score_logits", None) is not None:
        logits = outputs.object_score_logits.detach().cpu().flatten()
        if logits.numel():
            idx = min(obj_index, logits.numel() - 1)
            return float(torch.sigmoid(logits[idx]))

    return 0.75


def _mask_tensor_to_result(
    frame_idx: int,
    obj_id: int,
    masks,
    outputs,
    obj_index: int = 0,
) -> tuple[FrameMaskResult, np.ndarray]:
    import torch

    if isinstance(masks, torch.Tensor):
        m = masks.detach().cpu()
    else:
        m = torch.as_tensor(masks)

    if m.ndim == 4:
        idx = min(obj_index, m.shape[0] - 1)
        m = m[idx]
    if m.ndim == 3 and m.shape[0] > 1:
        if getattr(outputs, "iou_scores", None) is not None:
            scores = outputs.iou_scores.detach().cpu()
            while scores.ndim > 2:
                scores = scores[0]
            best_idx = int(torch.argmax(scores).item())
            confidence = float(scores.flatten()[best_idx])
        else:
            best_idx = 0
            confidence = _output_confidence(outputs, obj_index)
        mask_logits = m[best_idx]
    elif m.ndim == 3:
        mask_logits = m[0]
        confidence = _output_confidence(outputs, obj_index)
    elif m.ndim == 2:
        mask_logits = m
        confidence = _output_confidence(outputs, obj_index)
    else:
        raise ValueError(f"Unexpected mask tensor shape: {tuple(m.shape)}")

    mask = mask_logits.float().numpy() > 0.0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        empty = np.zeros_like(mask, dtype=bool)
        return (
            FrameMaskResult(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox=[0, 0, 0, 0],
                confidence=0.0,
                segmentation=[],
            ),
            empty,
        )
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    seg = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]
    return (
        FrameMaskResult(
            frame_idx=frame_idx,
            obj_id=obj_id,
            bbox=[x_min, y_min, x_max, y_max],
            confidence=confidence,
            segmentation=seg,
        ),
        mask,
    )


def _result_with_preview(result: FrameMaskResult, mask: np.ndarray) -> dict[str, Any]:
    payload = result.model_dump()
    payload["mask_preview_b64"] = _mask_to_b64(mask)
    return payload


def _mask_to_b64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _session_target_obj_ids(session: ChunkSession) -> list[int]:
    obj_ids = sorted({oid for by_obj in session.clicks.values() for oid in by_obj})
    return obj_ids or [1]


def _meta_pick_mask_result(
    frame_idx: int,
    outputs: dict[str, Any],
    target_obj_ids: list[int],
) -> tuple[FrameMaskResult, np.ndarray] | tuple[None, None]:
    for obj_id in target_obj_ids:
        result, mask = _meta_outputs_to_result(frame_idx, obj_id, outputs)
        if mask.any():
            return result, mask
    for oid in np.asarray(outputs.get("out_obj_ids", [])).reshape(-1).tolist():
        result, mask = _meta_outputs_to_result(frame_idx, int(oid), outputs)
        if mask.any():
            return result, mask
    return None, None


def _propagate_meta(session: ChunkSession, direction: str) -> Iterator[FrameMaskResult]:
    handle = session.inference_handle
    masks_dir = storage.chunk_masks_dir(session.upload_id, session.chunk_index)
    target_obj_ids = _session_target_obj_ids(session)

    if session.mode == "points":
        for frame_idx, by_obj in session.clicks.items():
            rel = _global_to_local(handle, frame_idx)
            for obj_id, clicks in by_obj.items():
                _meta_add_prompt(handle, rel, obj_id, clicks, clear_old_points=True)
    elif session.mode == "text" and session.text_prompt:
        for frame_idx in session.clicks:
            rel = _global_to_local(handle, frame_idx)
            _meta_add_prompt(
                handle,
                rel,
                obj_id=1,
                clicks=None,
                text=session.text_prompt,
                clear_old_points=True,
            )
            break

    stream = _meta_predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": handle["session_id"],
            "propagation_direction": direction,
        }
    )
    stems: list[int] = handle["frame_stems"]
    saved_frames: set[int] = set()
    yielded = 0
    for resp in stream:
        local_idx = int(resp["frame_index"])
        if local_idx < 0 or local_idx >= len(stems):
            continue
        abs_frame = stems[local_idx]
        if abs_frame < session.save_start or abs_frame > session.save_end:
            continue
        outputs = resp.get("outputs")
        if not outputs:
            continue
        picked = _meta_pick_mask_result(abs_frame, outputs, target_obj_ids)
        if picked[0] is None:
            continue
        result, mask = picked
        if abs_frame not in saved_frames:
            _save_mask(masks_dir, result, mask)
            saved_frames.add(abs_frame)
        yielded += 1
        yield result
    logger.info(
        "Meta propagate chunk %s: %s masks saved (%s frames yielded, direction=%s)",
        session.chunk_index,
        len(saved_frames),
        yielded,
        direction,
    )


def propagate_chunk(session: ChunkSession, direction: str = "both") -> Iterator[FrameMaskResult]:
    init_chunk_session(session)
    masks_dir = storage.chunk_masks_dir(session.upload_id, session.chunk_index)

    if session.engine_kind == "mock":
        meta = storage.load_upload_meta(session.upload_id)
        w = int(meta.get("width") or 640)
        h = int(meta.get("height") or 480)
        for frame_idx in range(session.process_start, session.process_end + 1):
            if frame_idx < session.save_start or frame_idx > session.save_end:
                continue
            result, mask = _mock_bbox(frame_idx, 1, w, h)
            _save_mask(masks_dir, result, mask)
            yield result
        return

    if session.engine_kind and session.engine_kind.startswith("meta_"):
        yield from _propagate_meta(session, direction)
        return

    import torch

    inference_session = session.inference_handle

    if session.mode == "points":
        for frame_idx, by_obj in session.clicks.items():
            rel = frame_idx - session.process_start
            for obj_id, clicks in by_obj.items():
                pts = [[[[p.x, p.y] for p in clicks]]]
                labels = [[[p.label for p in clicks]]]
                _tracker_processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=rel,
                    obj_ids=obj_id,
                    input_points=pts,
                    input_labels=labels,
                )
    elif session.mode == "text" and session.text_prompt:
        raise NotImplementedError("Text video mode wired in next iteration")

    reverse_passes: list[bool] = []
    if direction in ("forward", "both"):
        reverse_passes.append(False)
    if direction in ("backward", "both"):
        reverse_passes.append(True)
    if not reverse_passes:
        reverse_passes = [False]

    saved_frames: set[int] = set()
    for reverse in reverse_passes:
        for out in _tracker_model.propagate_in_video_iterator(inference_session, reverse=reverse):
            abs_frame = session.process_start + int(out.frame_idx)
            if abs_frame < session.save_start or abs_frame > session.save_end:
                continue
            vh, vw = _session_video_size(inference_session)
            masks = _tracker_processor.post_process_masks(
                [out.pred_masks.cpu()],
                original_sizes=[[vh, vw]],
                binarize=False,
            )[0]
            for obj_i, obj_id in enumerate(inference_session.obj_ids):
                result, mask = _mask_tensor_to_result(
                    abs_frame, int(obj_id), masks[obj_i : obj_i + 1], out, obj_index=0
                )
                if abs_frame not in saved_frames:
                    _save_mask(masks_dir, result, mask)
                    saved_frames.add(abs_frame)
                yield result


def _save_mask(masks_dir: Path, result: FrameMaskResult, mask: np.ndarray) -> None:
    path = masks_dir / f"{result.frame_idx:06d}_obj{result.obj_id}.json"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    png_path = masks_dir / f"{result.frame_idx:06d}_obj{result.obj_id}.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(png_path)

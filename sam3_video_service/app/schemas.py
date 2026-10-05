"""Request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = Field(..., description="1=positive, 0=negative")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 0 (negative) or 1 (positive)")
        return v


class ChunkInfo(BaseModel):
    chunk_index: int
    process_start: int
    process_end: int
    save_start: int
    save_end: int


class PrepareChunkRequest(BaseModel):
    chunk_index: int = Field(..., ge=0)


class FinalizeUploadRequest(BaseModel):
    original_filename: str = Field(..., min_length=1)


class PointsRequest(BaseModel):
    frame_idx: int = Field(..., ge=0)
    obj_id: int = Field(1, ge=1)
    points: list[PointPrompt] = Field(..., min_length=1)
    replace: bool = False


class TextPromptRequest(BaseModel):
    text: str = Field(..., min_length=1)
    frame_idx: int = Field(0, ge=0)


class RefineRequest(BaseModel):
    frame_idx: int = Field(..., ge=0)
    obj_id: int = Field(1, ge=1)
    mode: Literal["points", "text"] = "points"


class PropagateRequest(BaseModel):
    mode: Literal["points", "text"] = "points"
    direction: Literal["forward", "backward", "both"] = "both"


class FrameMaskResult(BaseModel):
    frame_idx: int
    obj_id: int
    bbox: list[int]
    confidence: float
    segmentation: list[list[int]] = Field(default_factory=list)


class RLEMask(BaseModel):
    size: list[int] = Field(..., min_length=2, max_length=2, description="[height, width]")
    counts: str | list[int] = Field(..., description="COCO compressed string or uncompressed run list")


class MaskInput(BaseModel):
    """Exactly one of png_b64, rle, polygons. Mask should match the video frame size."""

    png_b64: str | None = None
    rle: RLEMask | None = None
    polygons: list[list[float]] | None = Field(
        None, description="COCO polygons: flat [x1, y1, x2, y2, ...] lists in pixels"
    )

    @model_validator(mode="after")
    def exactly_one(self) -> "MaskInput":
        given = [v is not None for v in (self.png_b64, self.rle, self.polygons)]
        if sum(given) != 1:
            raise ValueError("provide exactly one of png_b64, rle, polygons")
        return self


class ObjectMaskPrompt(BaseModel):
    obj_id: int = Field(..., ge=1)
    label: str | None = Field(None, description="Class name, e.g. 'weld_pool'; used as COCO category")
    mask: MaskInput


class KeyframePrompt(BaseModel):
    frame_idx: int = Field(..., ge=0, description="0-based decoded frame index (see GET /uploads/{id}/frames/{idx}.jpg)")
    objects: list[ObjectMaskPrompt] = Field(..., min_length=1)


class TrackRequest(BaseModel):
    keyframes: list[KeyframePrompt] = Field(..., min_length=1)
    direction: Literal["forward", "both"] = "both"
    start_frame: int | None = Field(None, ge=0, description="Inclusive; defaults to 0")
    end_frame: int | None = Field(None, ge=0, description="Inclusive; defaults to last frame")


class JobProgressEvent(BaseModel):
    type: str
    frame_idx: int | None = None
    total: int | None = None
    message: str | None = None
    result: FrameMaskResult | None = None

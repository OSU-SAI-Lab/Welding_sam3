"""Binary mask codecs: PNG base64, COCO RLE (compressed + uncompressed), polygons."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw


class MaskDecodeError(ValueError):
    pass


def decode_png_b64(data: str) -> np.ndarray:
    """Any PNG (L, RGB, RGBA, 1-bit); non-zero pixels are foreground."""
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
    except Exception as e:
        raise MaskDecodeError(f"invalid PNG mask: {e}") from e
    if img.mode in ("RGBA", "LA"):
        return np.asarray(img)[..., -1] > 0
    return np.asarray(img.convert("L")) > 0


def encode_png_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_polygons(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """COCO-style polygons: each is a flat [x1, y1, x2, y2, ...] list."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) < 6 or len(poly) % 2:
            raise MaskDecodeError("each polygon needs >= 3 (x, y) pairs")
        draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=255)
    return np.asarray(img) > 0


# COCO RLE: counts of alternating 0/1 runs over the column-major (Fortran) flattened
# mask, starting with a run of zeros. The compressed string form is pycocotools'
# LEB128-like encoding (maskApi.c rleToString / rleFrString).


def mask_to_rle_counts(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return runs


def rle_counts_to_mask(counts: list[int], height: int, width: int) -> np.ndarray:
    if sum(counts) != height * width:
        raise MaskDecodeError(
            f"RLE counts sum to {sum(counts)}, expected {height * width} for size [{height}, {width}]"
        )
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    return flat.reshape((height, width), order="F")


def counts_to_string(counts: list[int]) -> str:
    out: list[str] = []
    for i, cnt in enumerate(counts):
        x = cnt - counts[i - 2] if i > 2 else cnt
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_counts(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            if p >= len(s):
                raise MaskDecodeError("truncated RLE string")
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(rle: dict) -> np.ndarray:
    try:
        height, width = (int(v) for v in rle["size"])
        counts = rle["counts"]
    except (KeyError, TypeError, ValueError) as e:
        raise MaskDecodeError("RLE needs 'size': [h, w] and 'counts'") from e
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if isinstance(counts, str):
        counts = string_to_counts(counts)
    return rle_counts_to_mask([int(c) for c in counts], height, width)


def encode_rle(mask: np.ndarray) -> dict:
    """Compressed COCO RLE, compatible with pycocotools.mask.decode."""
    h, w = mask.shape
    return {"size": [int(h), int(w)], "counts": counts_to_string(mask_to_rle_counts(mask))}


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]

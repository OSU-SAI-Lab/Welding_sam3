# SAM3 Video Labeler

Browser-based video labeling with Meta **SAM3**:

**upload → chunk → click points → preview mask → track → export MP4 / COCO / YOLO**

This repository evolved from an earlier SAM2 CLI point-prompt pipeline into a full **SAM3** web UI + GPU service for applied annotation (including welding video workflows).

## Demo results

Annotated welding overlay (CEE204 cut). GIF plays inline; full MP4 is linked below.

![Annotated welding demo](docs/images/welding_annotated_demo.gif)

**Full video (~34s, H.264, ~42 MB):**  
https://github.com/OSU-SAI-Lab/Welding_sam3/blob/main/docs/videos/welding_annotated_demo.mp4


## Layout

```text
sam3-video-labeler/
  sam3_video_service/   # GPU API (port 2129)
  labeler_ui/           # Wizard UI (port 8080)
  scripts/
    setup_env.sh        # create venv + install deps + clone Meta SAM3
    run_local.sh        # start API + UI
  docs/
    images/             # README demo GIF
    videos/             # annotated demo MP4
  data/                 # uploads + masks (gitignored)
  third_party/sam3/     # created by setup_env.sh (gitignored)
```

## Requirements

- Python **3.10+** (3.12 recommended)
- **ffmpeg** / `ffprobe` (video probe + export)
- NVIDIA GPU + CUDA for real SAM3 tracking (optional — mock mode works on CPU)
- Hugging Face access to [`facebook/sam3`](https://huggingface.co/facebook/sam3)

## One-time setup

```bash
# Personal repo
git clone https://github.com/Pratham-22/sam2-video-segmentation.git
cd sam2-video-segmentation

# Lab mirror (same codebase)
# git clone https://github.com/OSU-SAI-Lab/Welding_sam3.git
# cd Welding_sam3

chmod +x scripts/*.sh
./scripts/setup_env.sh
```

Then request model access and log in:

```bash
source .venv/bin/activate
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

## Run

```bash
# Optional: put large uploads on project storage
export DATA_ROOT=/path/to/large/disk/sam3-video-labeler-data
mkdir -p "$DATA_ROOT"

./scripts/run_local.sh
```

- **UI:** http://127.0.0.1:8080  
- **API health:** http://127.0.0.1:2129/health  

### NRP (Kubernetes) deployment

`sam3_video_service` (the GPU backend) can run standalone on NRP/Nautilus,
with the frontend (Smart Labeler) talking to it over HTTPS. See
[k8s/README.md](k8s/README.md) for the Dockerfile, manifests, and deploy
steps.

### SSH tunnel (cluster / remote GPU)

```bash
ssh -L 8080:localhost:8080 -L 2129:localhost:2129 user@gpu-host
```

Then open http://127.0.0.1:8080 on your laptop.

### OSC notes

```bash
# inside an interactive GPU allocation
module load python/3.12
module load ffmpeg/6.1.1
export DATA_ROOT=/fs/ess/PAS2699/$USER/sam3-video-labeler-data
mkdir -p "$DATA_ROOT"
./scripts/run_local.sh
```

## UI features

- Step-by-step wizard with time estimates
- Overall progress bar across chunks
- Drag-and-drop upload
- Positive / negative clicks, undo, delete points
- Preview mask on one frame before tracking
- Track chunk with elapsed + remaining time
- Export annotated MP4 when all chunks are done

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_ROOT` | `./data` | Uploads, masks, exports |
| `SAM3_MOCK` | auto (`0` on GPU, `1` otherwise) | `1` = fake masks for UI testing |
| `HF_TOKEN` | — | Hugging Face token for SAM3 weights |
| `SAM3_REPO` | `./third_party/sam3` | Path to Meta SAM3 checkout |
| `SAM3_BACKEND` | `transformers` | `transformers` or `meta` |
| `SAM3_VERSION` | `sam3` | `sam3` or `sam3.1` |
| `CHUNK_SIZE` | `1000` | Frames per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `VENV_DIR` | `./.venv` | Override venv location for `setup_env.sh` |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated origins allowed to call the API (set to the frontend's URL in production) |

## API (`sam3_video_service`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/uploads` | Upload video |
| GET | `/uploads/{id}/status` | Chunk progress |
| POST | `/uploads/{id}/chunks/prepare` | Extract frames + session |
| POST | `/sessions/{id}/prompt/points` | Positive / negative clicks |
| POST | `/sessions/{id}/refine` | Preview one frame |
| POST | `/sessions/{id}/propagate` | Track chunk (SSE job) |
| POST | `/uploads/{id}/export` | Build annotated MP4 |
| GET | `/uploads/{id}/export/download` | Download MP4 |
| POST | `/uploads/{id}/export/coco` | Build COCO JSON from masks |
| GET | `/uploads/{id}/export/coco/download` | Download COCO JSON |
| POST | `/uploads/{id}/export/yolo` | Build YOLO-style JSON from masks |
| GET | `/uploads/{id}/export/yolo/download` | Download YOLO JSON |
| POST | `/uploads/{id}/export/bbox-zip` | Build ZIP of frames with drawn boxes |
| GET | `/uploads/{id}/export/bbox-zip/download` | Download bbox images ZIP |

### Dataset exports

After at least one chunk is tracked, Step 5 can download:

- **COCO JSON** — `images` + `annotations` with `bbox` `[x, y, w, h]` and optional polygon `segmentation`
- **YOLO JSON** — per-frame objects with normalized `bbox_xywhn` (`cx cy w h`) ready to turn into `labels/*.txt`
- **BBox images ZIP** — JPEGs named like the original video stem, with green boxes drawn on tracked objects

### Multi-object tracking from keyframe masks

For clients that already produce masks (for example a smart labeler that runs point or text prompts per frame). Send masks for one or more objects on one or more keyframes. The service tracks every object across the whole video (or a frame range), chunk by chunk, forward and backward.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/uploads` | Upload video (same as above) → `upload_id`, `frame_count`, `width`, `height` |
| GET | `/uploads/{id}/frames/{frame_idx}.jpg` | Frame exactly as the tracker indexes it |
| POST | `/uploads/{id}/track` | Start a tracking job from keyframe masks → `job_id` |
| GET | `/track-jobs/{job_id}` | Status: `queued` / `running` / `done` / `failed` / `cancelled`, progress |
| GET | `/track-jobs/{job_id}/events` | SSE stream of status updates until the job ends |
| DELETE | `/track-jobs/{job_id}` | Cancel |
| GET | `/track-jobs/{job_id}/frames/{frame_idx}` | All objects on one frame: COCO RLE, bbox, area (`?include_png=true` adds PNG base64) |
| GET | `/track-jobs/{job_id}/masks/{frame_idx}/{obj_id}.png` | One mask as PNG |
| GET | `/track-jobs/{job_id}/coco` | COCO JSON, RLE segmentation, `track_id` = `obj_id`, categories from labels |
| GET | `/track-jobs/{job_id}/video` | Annotated MP4, one color per object (rendered on first request) |

**Frame indexing:** `frame_idx` is the 0-based index of the decoded frame, one per real frame, with no frame-rate resampling. Masks must be drawn on these frames. Use `/uploads/{id}/frames/{idx}.jpg`, or decode with `ffmpeg -vsync 0`. Variable-frame-rate video such as screen recordings will be misaligned if frames are sampled by timestamp.

`POST /uploads/{id}/track` body:

```json
{
  "keyframes": [
    {
      "frame_idx": 120,
      "objects": [
        {"obj_id": 1, "label": "torch", "mask": {"rle": {"size": [726, 1284], "counts": "<COCO RLE string>"}}},
        {"obj_id": 2, "label": "weld_pool", "mask": {"png_b64": "<base64 PNG, non-zero = object>"}}
      ]
    },
    {
      "frame_idx": 900,
      "objects": [
        {"obj_id": 3, "label": "wire", "mask": {"polygons": [[10, 20, 60, 20, 60, 80, 10, 80]]}}
      ]
    }
  ],
  "direction": "both",
  "start_frame": null,
  "end_frame": null
}
```

- Each mask takes exactly one of `png_b64`, `rle` (COCO, compressed string or uncompressed list), or `polygons`, at the video's frame size.
- Keep the same `obj_id` for the same object across keyframes. Extra keyframes for an object re-anchor it, which helps after occlusion or drift.
- Objects may first appear on different keyframes. With `direction: "both"`, each object is also tracked backward from its first keyframe to `start_frame`.
- Jobs run one at a time on the GPU. Results are written to `DATA_ROOT/track_jobs/{job_id}/` and survive a service restart.
- Requires `SAM3_BACKEND=transformers`. With `SAM3_MOCK=1`, the nearest keyframe mask is copied to every frame, for API testing.


## Troubleshooting

| Issue | Fix |
|-------|-----|
| Disk quota exceeded on upload | Set `DATA_ROOT` to a large disk |
| `ffprobe not found` | Install ffmpeg / `module load ffmpeg` |
| Stuck loading SAM3 | First track loads weights (~1–3 min) |
| Mock masks only | GPU available? Set `SAM3_MOCK=0` and log in to Hugging Face |
| `No venv found` | Run `./scripts/setup_env.sh` |

## License / third-party

- Meta SAM3 (`facebookresearch/sam3`) and Hugging Face weights have their own licenses and access terms — request access before using real tracking.

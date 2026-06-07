# AI / Codex Context

This document is for future AI/Codex sessions. The user-facing README is intentionally written in Chinese.

## Project

Purpose:

- Local RF-DETR and YOLO object detection/tracking validation.
- Export per-frame tracking data.
- Preview visual overlays in a small browser player.
- Keep live camera detection in the separate `rfdetr-live` project.
- Keep this separate from the older OpenCV.js contour-detection project.

## Environment Policy

Do not document machine-specific local paths, installed models, or GPU hardware as repository facts.

Recommended local environments:

- `video-track-cpu-env`: local CPU fallback, usually created with `venv`.
- `video-track-gpu-env`: local NVIDIA CUDA environment, usually created with `conda` and Python 3.11.

These environment folders must be ignored by Git.

## Main Script

```text
scripts/track_objects_rfdetr.py
scripts/track_objects_yolo.py
```

Behavior:

- Defaults to newest `.mp4` inside `source/`.
- YOLO script uses `YOLO(...).track(...)`, not `predict(...)`.
- RF-DETR script uses `model.predict(...)` per frame and `supervision.ByteTrack` for track IDs.
- Default model is `yolo11n.pt`.
- Default RF-DETR size is `medium`.
- Default class is `car`.
- Default confidence is `0.15`.
- Default RF-DETR confidence is `0.35`.
- Supports `--device`, e.g. `cpu`, `0`, `mps`.
- Outputs one run folder per execution under `runs/`.
- Run folder name includes model, class label, device label, confidence, timestamp.
- Copies the original video into the run folder using its original filename.
- Generated output files include the source stem.

Output example:

```text
runs/yolo11m_person_gpu_conf0p15_20260521_180000/
  input_video.mp4
  input_video_stickers.mp4
  input_video_tracks.csv
  input_video_tracks.jsonl
```

JSONL format:

```json
{"frame": 1, "objects": [...]}
{"frame": 2, "objects": [...]}
```

Important fields:

- `track_id`
- `class_id`
- `class_name`
- `confidence`
- `x1, y1, x2, y2`
- `center_x, center_y`
- `center_x_norm, center_y_norm`

## Frontend

Path:

```text
web_track_overlay_player/
```

Files:

```text
index.html
style.css
app.js
```

Behavior:

- Drag/drop original video and generated `*_tracks.jsonl`.
- No native video controls.
- Playback controlled by a panel button.
- Video fills the window using cover behavior.
- Canvas overlay uses the same cover transform, so detection coordinates stay aligned after scaling/cropping.
- Supports filled boxes, ellipses, and center dots.

Must be served over local HTTP, not opened via `file://`.

## Known Directions

The user is considering MIDI output based on detections. Keep it modular:

```text
tracks.jsonl -> event engine -> MIDI mapper -> Web MIDI output
```

Avoid sending MIDI every frame for every box. Prefer stable events:

- `track_enter`
- `track_exit`
- `zone_enter`
- `zone_leave`
- rate-limited CC messages for position/area/count

## Coding Preferences

- Keep generated outputs under `runs/`.
- Do not commit model weights, videos, run outputs, or virtual environments.
- Dependency files are `requirements.txt`, `requirements-cpu.txt`, and `requirements-gpu-cu121.txt`.
- Prefer explicit CLI arguments over hidden behavior.
- Keep the Python detection workflow and frontend visual experiment loosely coupled.
- If adding GitHub support, include `.gitignore`, `requirements` files, and this context document.

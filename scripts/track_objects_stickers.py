import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO


DEFAULT_CLASS_IDS = [2]
CLASS_ALIASES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "bird": 14,
    "brid": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
}


def pick_default_video(root: Path) -> Path:
    source_dir = root / "source"
    search_dir = source_dir if source_dir.exists() else root
    videos = sorted(search_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        raise FileNotFoundError(
            "No .mp4 file found in ./source. Pass a video path with --source."
        )
    return videos[0]


def draw_sticker(frame, track_id, class_name, box, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"{class_name.upper()} {track_id}"
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_w = text_size[0] + 18
    label_h = text_size[1] + baseline + 14

    sticker_x = max(0, min(cx - label_w // 2, frame.shape[1] - label_w))
    sticker_y = max(0, y1 - label_h - 10)

    cv2.rectangle(
        frame,
        (sticker_x, sticker_y),
        (sticker_x + label_w, sticker_y + label_h),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (sticker_x + 9, sticker_y + label_h - baseline - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (12, 12, 12),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)


def color_for_id(track_id):
    seed = int(track_id) * 37
    return (
        80 + (seed * 3) % 176,
        80 + (seed * 5) % 176,
        80 + (seed * 7) % 176,
    )


def bgr_to_hex(color):
    blue, green, red = [int(v) for v in color]
    return f"#{red:02x}{green:02x}{blue:02x}"


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def parse_classes(value):
    if not value:
        return DEFAULT_CLASS_IDS

    class_ids = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key in CLASS_ALIASES:
            class_ids.append(CLASS_ALIASES[key])
        else:
            try:
                class_ids.append(int(key))
            except ValueError as exc:
                known = ", ".join(sorted(CLASS_ALIASES))
                raise ValueError(
                    f"Unknown class '{part}'. Use a numeric COCO class ID or one of: {known}"
                ) from exc
    return class_ids or DEFAULT_CLASS_IDS


def classes_label(class_ids, names):
    labels = []
    for class_id in class_ids:
        labels.append(safe_name(names.get(class_id, str(class_id))))
    return "-".join(labels)


def device_label(device):
    if device is not None:
        value = str(device).lower()
        if value == "cpu":
            return "cpu"
        if value == "mps":
            return "mps"
        return "gpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Track YOLO objects and draw ID stickers.")
    parser.add_argument("--source", type=Path, default=None, help="Input video path.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path.")
    parser.add_argument(
        "--classes",
        default="car",
        help="Comma-separated class IDs or aliases, e.g. car, person,car, 0,2.",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, 0 for NVIDIA CUDA GPU, or mps on Apple Silicon.",
    )
    parser.add_argument("--show", action="store_true", help="Preview while processing.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N frames. Use 0 to disable.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    source = args.source if args.source else pick_default_video(root)
    if not source.is_absolute():
        source = root / source

    model = YOLO(args.model)
    class_ids = parse_classes(args.classes)
    class_label = classes_label(class_ids, model.names)
    model_name = safe_name(Path(args.model).stem)
    conf_name = str(args.conf).replace(".", "p")
    device_name = device_label(args.device)
    run_name = (
        f"{model_name}_{class_label}_{device_name}_conf{conf_name}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_name = safe_name(source.stem)
    out_video = run_dir / f"{source_name}_stickers.mp4"
    out_jsonl = run_dir / f"{source_name}_tracks.jsonl"
    out_csv = run_dir / f"{source_name}_tracks.csv"

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    results = model.track(
        source=str(source),
        classes=class_ids,
        conf=args.conf,
        device=args.device,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    csv_fields = [
        "frame",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "color",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",
    ]

    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as cf:
        csv_writer = csv.DictWriter(cf, fieldnames=csv_fields)
        csv_writer.writeheader()

        start_time = perf_counter()
        for frame_index, result in enumerate(results, start=1):
            frame = result.orig_img.copy()
            boxes = result.boxes
            frame_records = []

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for box, track_id, conf, cls in zip(xyxy, ids, confs, classes):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    color = color_for_id(track_id)
                    color_hex = bgr_to_hex(color)
                    object_name = model.names.get(int(cls), str(cls))
                    draw_sticker(frame, track_id, object_name, box, color)

                    record = {
                        "frame": frame_index,
                        "track_id": int(track_id),
                        "class_id": int(cls),
                        "class_name": object_name,
                        "confidence": round(float(conf), 4),
                        "color": color_hex,
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "center_x": round(cx, 2),
                        "center_y": round(cy, 2),
                        "center_x_norm": round(cx / width, 6),
                        "center_y_norm": round(cy / height, 6),
                    }
                    frame_records.append(record)
                    csv_writer.writerow(record)

            jf.write(json.dumps({"frame": frame_index, "objects": frame_records}) + "\n")
            writer.write(frame)

            if args.progress_every and frame_index % args.progress_every == 0:
                elapsed = max(perf_counter() - start_time, 0.001)
                process_fps = frame_index / elapsed
                if total_frames:
                    percent = frame_index / total_frames * 100
                    progress = f"{frame_index}/{total_frames} ({percent:5.1f}%)"
                else:
                    progress = f"{frame_index} frames"
                print(
                    f"\rProcessing {progress} | {process_fps:5.1f} fps | "
                    f"objects in frame: {len(frame_records)}",
                    end="",
                    flush=True,
                )

            if args.show:
                cv2.imshow("car stickers", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    writer.release()
    cv2.destroyAllWindows()
    print()

    print(f"Video: {out_video}")
    print(f"JSONL: {out_jsonl}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()

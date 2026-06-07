import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch


DEFAULT_CLASS_IDS = [3]
CLASS_ALIASES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "bird": 16,
    "brid": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
}

COCO_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

RFDETR_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
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


def draw_sticker(frame, track_id, class_name, box, color, draw_box=True):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    box_scale = min(box_w, box_h)
    text_scale = max(0.35, min(0.7, box_scale / 120))
    thickness = max(1, min(2, int(round(box_scale / 80))))
    pad_x = max(5, int(round(9 * text_scale / 0.7)))
    pad_y = max(4, int(round(7 * text_scale / 0.7)))
    gap = max(3, int(round(10 * text_scale / 0.7)))
    dot_radius = max(3, min(6, int(round(box_scale / 30))))

    if draw_box:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label = f"{class_name.upper()} {track_id}"
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    label_w = text_size[0] + pad_x * 2
    label_h = text_size[1] + baseline + pad_y * 2

    sticker_x = max(0, min(cx - label_w // 2, frame.shape[1] - label_w))
    sticker_y = max(0, y1 - label_h - gap)

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
        (sticker_x + pad_x, sticker_y + label_h - baseline - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (12, 12, 12),
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)


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
    if value.strip().lower() == "all":
        return None

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
    if class_ids is None:
        return "all"
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


def create_rfdetr_model(size, device):
    import rfdetr

    class_name = RFDETR_MODEL_CLASSES[size]
    model_class = getattr(rfdetr, class_name)
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    return model_class(**kwargs)


def optimize_model_for_inference(model):
    optimize = getattr(model, "optimize_for_inference", None)
    if optimize is None:
        return
    optimize()


def create_tracker(frame_rate):
    import supervision as sv

    try:
        return sv.ByteTrack(frame_rate=frame_rate)
    except TypeError:
        return sv.ByteTrack()


def detections_for_frame(model, frame, threshold, class_ids):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(rgb, threshold=threshold)
    if class_ids and getattr(detections, "class_id", None) is not None:
        class_mask = np.isin(detections.class_id, class_ids)
        detections = detections[class_mask]
    return detections


def apply_nms(detections, threshold):
    if threshold <= 0:
        return detections
    with_nms = getattr(detections, "with_nms", None)
    if with_nms is None:
        return detections
    return with_nms(threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="Track RF-DETR objects and draw ID stickers.")
    parser.add_argument("--source", type=Path, default=None, help="Input video path.")
    parser.add_argument(
        "--model-size",
        choices=sorted(RFDETR_MODEL_CLASSES),
        default="medium",
        help="RF-DETR model size. These default options use the Apache 2.0 model line.",
    )
    parser.add_argument(
        "--classes",
        default="car",
        help="Comma-separated RF-DETR COCO category IDs or aliases, e.g. car, person,car, 1,3. Use all to disable class filtering.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for duplicate-box suppression. Use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device when supported by the installed RF-DETR package, e.g. cpu or cuda.",
    )
    parser.add_argument("--show", action="store_true", help="Preview while processing.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip RF-DETR optimize_for_inference().",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--debug-detections",
        type=int,
        default=0,
        help="Print raw RF-DETR detections before class filtering for the first N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to process the full video.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    source = args.source if args.source else pick_default_video(root)
    if not source.is_absolute():
        source = root / source

    class_ids = parse_classes(args.classes)
    class_label = classes_label(class_ids, COCO_NAMES)
    model_name = safe_name(f"rfdetr-{args.model_size}")
    conf_name = str(args.conf).replace(".", "p")
    device_name = device_label(args.device)
    run_name = (
        f"{model_name}_{class_label}_{device_name}_conf{conf_name}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_name = safe_name(source.stem)
    original_video = run_dir / source.name
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
    frame_rate = max(1, int(round(fps)))
    shutil.copy2(source, original_video)

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    print(
        f"Loading RF-DETR {args.model_size} on {args.device or 'auto device'}...",
        flush=True,
    )
    model = create_rfdetr_model(args.model_size, args.device)
    if not args.no_optimize:
        print("Optimizing RF-DETR for inference. This can take a while on the first run...", flush=True)
        optimize_model_for_inference(model)
        print("Optimization complete.", flush=True)
    else:
        print("Skipping RF-DETR inference optimization.", flush=True)
    tracker = create_tracker(frame_rate)

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
        "mask_polygon",
    ]

    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as cf:
        csv_writer = csv.DictWriter(cf, fieldnames=csv_fields)
        csv_writer.writeheader()

        start_time = perf_counter()
        frame_index = 0
        print(f"Processing video: {source} ({total_frames or 'unknown'} frames)", flush=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            raw_detections = detections_for_frame(model, frame, args.conf, None)
            raw_detections = apply_nms(raw_detections, args.nms_threshold)
            if args.debug_detections and frame_index <= args.debug_detections:
                raw_classes = getattr(raw_detections, "class_id", np.array([], dtype=int))
                raw_confs = getattr(raw_detections, "confidence", np.array([], dtype=float))
                debug_items = [
                    f"{COCO_NAMES.get(int(cls), str(cls))}:{int(cls)}:{float(conf):.3f}"
                    for cls, conf in zip(raw_classes, raw_confs)
                ]
                print(f"\nFrame {frame_index} raw detections: {', '.join(debug_items) or 'none'}")
            detections = raw_detections
            if class_ids and getattr(detections, "class_id", None) is not None:
                class_mask = np.isin(detections.class_id, class_ids)
                detections = detections[class_mask]
            detections = tracker.update_with_detections(detections)
            frame_records = []

            xyxy = getattr(detections, "xyxy", np.empty((0, 4)))
            confidences = getattr(detections, "confidence", None)
            detected_classes = getattr(detections, "class_id", None)
            tracker_ids = getattr(detections, "tracker_id", None)
            if confidences is None:
                confidences = np.ones(len(xyxy), dtype=float)
            if detected_classes is None:
                detected_classes = np.zeros(len(xyxy), dtype=int)
            if tracker_ids is None:
                tracker_ids = np.arange(1, len(xyxy) + 1)

            for box, track_id, conf, cls in zip(
                xyxy, tracker_ids, confidences, detected_classes
            ):
                if track_id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                color = color_for_id(track_id)
                color_hex = bgr_to_hex(color)
                object_name = COCO_NAMES.get(int(cls), str(cls))
                draw_sticker(frame, int(track_id), object_name, box, color)

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
                    "mask_polygon": "",
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
                cv2.imshow("RF-DETR stickers", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print()

    print(f"Original: {original_video}")
    print(f"Video: {out_video}")
    print(f"JSONL: {out_jsonl}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()

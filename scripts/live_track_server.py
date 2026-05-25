import argparse
import json
import platform
import re
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
from ultralytics import YOLO

from track_objects_stickers import (
    CLASS_ALIASES,
    bgr_to_hex,
    color_for_id,
    draw_sticker,
    parse_classes,
)


HTML_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>YOLO Live Track</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        background: #101312;
        color: #f4f7f5;
        font-family: Segoe UI, system-ui, sans-serif;
        overflow: hidden;
      }
      .stage {
        position: fixed;
        inset: 0;
        background: #080a09;
      }
      #stream, #overlay {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      #overlay { pointer-events: none; }
      .hud {
        position: fixed;
        top: 14px;
        right: 14px;
        display: grid;
        gap: 8px;
        min-width: 170px;
        padding: 12px;
        background: rgba(12, 14, 13, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        backdrop-filter: blur(10px);
      }
      .hud div {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        font-size: 13px;
      }
      .hud span { color: #aab6b0; }
      .hud strong { font-variant-numeric: tabular-nums; }
      .status {
        position: fixed;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        padding: 10px 14px;
        max-width: min(440px, calc(100vw - 32px));
        color: #dce6e1;
        background: rgba(12, 14, 13, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        text-align: center;
      }
      .status.is-hidden { display: none; }
      .control-panel {
        position: fixed;
        left: 14px;
        top: 14px;
        display: grid;
        gap: 8px;
        width: min(340px, calc(100vw - 28px));
        padding: 12px;
        background: rgba(12, 14, 13, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        backdrop-filter: blur(10px);
      }
      .control-panel header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
      }
      .control-panel strong { font-size: 14px; }
      .control-panel button {
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 6px;
        padding: 7px 9px;
        color: #f4f7f5;
        background: rgba(255, 255, 255, 0.08);
        font: inherit;
        cursor: pointer;
      }
      .field {
        display: grid;
        gap: 5px;
        color: #aab6b0;
        font-size: 12px;
      }
      .field select, .field input {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 6px;
        padding: 7px 9px;
        color: #f4f7f5;
        background: #1d211f;
        font: inherit;
      }
      .field option {
        color: #101312;
        background: #f4f7f5;
      }
      .field small {
        min-height: 16px;
        color: #c2ccc7;
        font-size: 12px;
      }
      @media (max-width: 700px) {
        .hud {
          left: 10px;
          right: 10px;
          top: auto;
          bottom: 10px;
          grid-template-columns: repeat(4, 1fr);
        }
        .hud div {
          display: grid;
          gap: 2px;
        }
      }
    </style>
  </head>
  <body>
    <main class="stage">
      <img id="stream" src="/video" alt="Live camera stream" />
      <canvas id="overlay"></canvas>
    </main>
    <section class="control-panel">
      <header>
        <strong>Live Controls</strong>
        <button id="scanCameras" type="button">Scan</button>
      </header>
      <label class="field">
        Camera
        <select id="cameraSelect"></select>
        <small id="cameraDetail">Scan cameras to choose a source.</small>
      </label>
      <label class="field">
        Class
        <select id="classSelect"></select>
      </label>
      <label id="customClassWrap" class="field" hidden>
        Custom classes
        <input id="customClassInput" type="text" value="person" />
      </label>
    </section>
    <div id="status" class="status">Waiting for camera frames...</div>
    <aside class="hud">
      <div><span>Frame</span><strong id="frameStat">-</strong></div>
      <div><span>Objects</span><strong id="objectStat">-</strong></div>
      <div><span>FPS</span><strong id="fpsStat">-</strong></div>
      <div><span>Size</span><strong id="sizeStat">-</strong></div>
    </aside>
    <script>
      const img = document.querySelector("#stream");
      const canvas = document.querySelector("#overlay");
      const ctx = canvas.getContext("2d");
      const frameStat = document.querySelector("#frameStat");
      const objectStat = document.querySelector("#objectStat");
      const fpsStat = document.querySelector("#fpsStat");
      const sizeStat = document.querySelector("#sizeStat");
      const status = document.querySelector("#status");
      const scanCameras = document.querySelector("#scanCameras");
      const cameraSelect = document.querySelector("#cameraSelect");
      const cameraDetail = document.querySelector("#cameraDetail");
      const classSelect = document.querySelector("#classSelect");
      const customClassWrap = document.querySelector("#customClassWrap");
      const customClassInput = document.querySelector("#customClassInput");
      let latest = { width: 1, height: 1, objects: [] };
      let streamNaturalWidth = 1;
      let streamNaturalHeight = 1;

      function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        const pw = Math.floor(w * dpr);
        const ph = Math.floor(h * dpr);
        if (canvas.width !== pw || canvas.height !== ph) {
          canvas.width = pw;
          canvas.height = ph;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        return { w, h };
      }

      function fitRect(w, h, sourceW, sourceH) {
        const sourceRatio = sourceW / sourceH;
        const targetRatio = w / h;
        if (targetRatio > sourceRatio) {
          const width = w;
          const height = width / sourceRatio;
          return { x: 0, y: (h - height) / 2, width, height, scale: width / sourceW };
        }
        const height = h;
        const width = height * sourceRatio;
        return { x: (w - width) / 2, y: 0, width, height, scale: width / sourceW };
      }

      function colorWithAlpha(color, alpha) {
        const match = String(color || "").match(/^#?([0-9a-f]{6})$/i);
        if (!match) return `rgba(80, 220, 150, ${alpha})`;
        const hex = match[1];
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      }

      function draw() {
        const size = resizeCanvas();
        ctx.clearRect(0, 0, size.w, size.h);
        const sourceW = latest.width || streamNaturalWidth || 1;
        const sourceH = latest.height || streamNaturalHeight || 1;
        const fit = fitRect(size.w, size.h, sourceW, sourceH);

        for (const obj of latest.objects || []) {
          const x = fit.x + obj.x1 * fit.scale;
          const y = fit.y + obj.y1 * fit.scale;
          const w = (obj.x2 - obj.x1) * fit.scale;
          const h = (obj.y2 - obj.y1) * fit.scale;
          const color = colorWithAlpha(obj.color, 0.95);
          const fill = colorWithAlpha(obj.color, 0.2);
          ctx.lineWidth = 2;
          ctx.strokeStyle = color;
          ctx.fillStyle = fill;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x, y, w, h);

          const label = `${obj.class_name || "obj"} ${obj.track_id}`;
          ctx.font = "700 13px Segoe UI, sans-serif";
          const labelW = Math.ceil(ctx.measureText(label).width) + 14;
          const labelH = 24;
          const labelX = Math.max(0, Math.min(x, size.w - labelW));
          const labelY = Math.max(0, y - labelH);
          ctx.fillStyle = color;
          ctx.fillRect(labelX, labelY, labelW, labelH);
          ctx.fillStyle = "#050706";
          ctx.textBaseline = "middle";
          ctx.fillText(label, labelX + 7, labelY + labelH / 2);
        }
        requestAnimationFrame(draw);
      }

      const events = new EventSource("/events");
      events.onmessage = (event) => {
        latest = JSON.parse(event.data);
        if (latest.status) {
          status.textContent = latest.status;
          status.classList.remove("is-hidden");
        } else {
          status.classList.add("is-hidden");
        }
        frameStat.textContent = latest.frame || "-";
        objectStat.textContent = (latest.objects || []).length;
        fpsStat.textContent = latest.process_fps ? latest.process_fps.toFixed(1) : "-";
        sizeStat.textContent = latest.width && latest.height ? `${latest.width}x${latest.height}` : "-";
      };
      events.onerror = () => {
        status.textContent = "Waiting for live tracking server...";
        status.classList.remove("is-hidden");
      };
      img.onload = () => {
        streamNaturalWidth = img.naturalWidth || streamNaturalWidth;
        streamNaturalHeight = img.naturalHeight || streamNaturalHeight;
      };
      window.addEventListener("resize", () => {
        resizeCanvas();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      });
      async function refreshCameras() {
        scanCameras.disabled = true;
        cameraSelect.innerHTML = "";
        const loading = document.createElement("option");
        loading.textContent = "Scanning...";
        loading.value = "";
        cameraSelect.appendChild(loading);
        cameraDetail.textContent = "Scanning camera devices...";
        try {
          const response = await fetch("/cameras");
          const data = await response.json();
          cameraSelect.innerHTML = "";
          const placeholder = document.createElement("option");
          placeholder.textContent = "Choose camera";
          placeholder.value = "";
          cameraSelect.appendChild(placeholder);
          for (const camera of data.cameras || []) {
            const detail = camera.width
              ? `${camera.width}x${camera.height} · brightness ${camera.brightness.toFixed(1)} · ${camera.status}`
              : camera.status;
            const option = document.createElement("option");
            option.value = `${camera.source}|${camera.backend}`;
            option.textContent = camera.name || `Camera ${camera.source}`;
            option.dataset.detail = `Source ${camera.source} · ${camera.backend} · ${detail}`;
            cameraSelect.appendChild(option);
          }
          if (cameraSelect.options.length === 1) cameraDetail.textContent = "No camera found.";
          else cameraDetail.textContent = "Choose a camera source.";
        } catch (error) {
          cameraSelect.innerHTML = "";
          cameraDetail.textContent = "Camera scan failed.";
        } finally {
          scanCameras.disabled = false;
        }
      }
      scanCameras.onclick = refreshCameras;
      cameraSelect.onchange = async () => {
        if (!cameraSelect.value) return;
        const [source, backend] = cameraSelect.value.split("|");
        const selected = cameraSelect.selectedOptions[0];
        cameraDetail.textContent = selected?.dataset.detail || "";
        status.textContent = `Switching to ${selected?.textContent || "camera"}...`;
        status.classList.remove("is-hidden");
        await fetch(`/select-camera?source=${source}&backend=${backend}`);
      };
      function currentClasses() {
        return classSelect.value === "custom" ? customClassInput.value : classSelect.value;
      }
      async function loadClasses() {
        try {
          const response = await fetch("/classes");
          const data = await response.json();
          const current = data.classes || "person";
          const options = data.options || [];
          classSelect.innerHTML = "";
          for (const name of options) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            classSelect.appendChild(option);
          }
          const combo = document.createElement("option");
          combo.value = "person,car";
          combo.textContent = "person,car";
          classSelect.appendChild(combo);
          const custom = document.createElement("option");
          custom.value = "custom";
          custom.textContent = "custom";
          classSelect.appendChild(custom);
          if (options.includes(current) || current === "person,car") {
            classSelect.value = current;
          } else {
            classSelect.value = "custom";
            customClassInput.value = current;
          }
          customClassWrap.hidden = classSelect.value !== "custom";
        } catch (error) {
          classSelect.value = "person";
        }
      }
      async function updateClasses() {
        customClassWrap.hidden = classSelect.value !== "custom";
        const classes = currentClasses().trim();
        if (!classes) return;
        status.textContent = `Tracking classes: ${classes}`;
        status.classList.remove("is-hidden");
        await fetch(`/set-classes?classes=${encodeURIComponent(classes)}`);
      }
      classSelect.onchange = updateClasses;
      customClassInput.onchange = updateClasses;
      customClassInput.onkeydown = (event) => {
        if (event.key === "Enter") updateClasses();
      };
      refreshCameras();
      loadClasses();
      requestAnimationFrame(draw);
    </script>
  </body>
</html>
"""


class LiveState:
    def __init__(self):
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.jpeg = None
        self.payload = None
        self.seq = 0
        self.camera_request = None
        self.current_camera = None
        self.classes_text = "person"
        self.class_ids = parse_classes("person")

    def publish(self, jpeg, payload):
        with self.condition:
            self.jpeg = jpeg
            self.payload = payload
            self.seq += 1
            self.condition.notify_all()

    def publish_status(self, status):
        with self.condition:
            self.payload = {
                "frame": 0,
                "timestamp": time.time(),
                "width": 1,
                "height": 1,
                "process_fps": 0,
                "status": status,
                "objects": [],
            }
            self.seq += 1
            self.condition.notify_all()

    def request_camera(self, source, backend):
        with self.condition:
            self.camera_request = (source, backend)
            self.payload = {
                "frame": 0,
                "timestamp": time.time(),
                "width": 1,
                "height": 1,
                "process_fps": 0,
                "status": f"Switching to camera source={source}, backend={backend}...",
                "objects": [],
            }
            self.seq += 1
            self.condition.notify_all()

    def take_camera_request(self):
        with self.condition:
            request = self.camera_request
            self.camera_request = None
            return request

    def set_current_camera(self, source, backend):
        with self.condition:
            self.current_camera = {"source": source, "backend": backend}

    def set_classes(self, classes_text):
        normalized = normalize_classes(classes_text)
        class_ids = parse_classes(normalized)
        with self.condition:
            self.classes_text = normalized
            self.class_ids = class_ids
            self.payload = {
                "frame": 0,
                "timestamp": time.time(),
                "width": 1,
                "height": 1,
                "process_fps": 0,
                "status": f"Tracking classes: {normalized}",
                "objects": [],
            }
            self.seq += 1
            self.condition.notify_all()
        return normalized, class_ids

    def get_class_ids(self):
        with self.condition:
            return list(self.class_ids)


def normalize_classes(classes_text):
    aliases = {
        "brid": "bird",
    }
    parts = []
    for part in str(classes_text or "person").split(","):
        key = part.strip().lower()
        parts.append(aliases.get(key, key))
    return ",".join(part for part in parts if part) or "person"


def parse_source(value):
    text = str(value)
    if text.lower() in {"none", "off"}:
        return "none"
    if text.lower() == "auto":
        return "auto"
    return int(text) if text.isdigit() else text


def camera_backend(name, source):
    key = str(name or "auto").lower()
    if key == "auto":
        if isinstance(source, int) and platform.system().lower() == "windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    backends = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "ffmpeg": cv2.CAP_FFMPEG,
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
    }
    if key not in backends:
        known = ", ".join(sorted(backends.keys() | {"auto"}))
        raise ValueError(f"Unknown camera backend '{name}'. Use one of: {known}")
    return backends[key]


def read_probe_frame(cap, attempts=30, delay=0.03):
    best_frame = None
    best_brightness = -1.0
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok:
            brightness = float(frame.mean())
            if brightness >= best_brightness:
                best_frame = frame
                best_brightness = brightness
        time.sleep(delay)
    return best_frame


def open_capture(args, state=None):
    source = parse_source(args.source)
    if source == "none":
        if state is not None:
            state.publish_status("Select a camera from the Cameras panel.")
        return cv2.VideoCapture(), "none", cv2.CAP_ANY
    if source == "auto":
        return open_auto_capture(args, state)

    backend = camera_backend(args.backend, source)
    if state is not None:
        state.publish_status(f"Opening camera source={source}, backend={args.backend}...")
    cap = cv2.VideoCapture(source, backend)
    configure_capture(cap, args, source)
    return cap, source, backend


def open_requested_capture(args, source, backend_name, state=None):
    previous_source = args.source
    previous_backend = args.backend
    args.source = str(source)
    args.backend = backend_name
    try:
        return open_capture(args, state)
    finally:
        args.source = previous_source
        args.backend = previous_backend


def configure_capture(cap, args, source=None):
    if args.width or args.height:
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    elif args.auto_resolution and isinstance(source, int):
        choose_best_capture_resolution(cap, args)

    if args.camera_fps:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)


def choose_best_capture_resolution(cap, args):
    candidates = resolution_candidates(args.resolution_mode)
    best = None

    for width, height in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = read_probe_frame(cap, attempts=4, delay=0.01)
        if frame is None:
            continue
        actual_h, actual_w = frame.shape[:2]
        area = actual_w * actual_h
        target_area = resolution_target_area(args.resolution_mode)
        if area > target_area * 1.15:
            continue
        if best is None or area > best[0]:
            best = (area, actual_w, actual_h)

    if best is not None:
        _, width, height = best
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def resolution_target_area(mode):
    if mode == "quality":
        return 1920 * 1080
    if mode == "speed":
        return 960 * 540
    return 1280 * 720


def resolution_candidates(mode):
    if mode == "quality":
        return [
            (1920, 1080),
            (1600, 1200),
            (1600, 900),
            (1440, 1080),
            (1280, 1024),
            (1280, 960),
            (1280, 720),
            (1024, 768),
            (1024, 576),
            (640, 480),
        ]
    if mode == "speed":
        return [
            (960, 720),
            (960, 540),
            (854, 480),
            (800, 600),
            (640, 480),
            (640, 360),
        ]
    return [
        (1280, 960),
        (1280, 720),
        (1024, 768),
        (1024, 576),
        (960, 720),
        (960, 540),
        (854, 480),
        (800, 600),
        (640, 480),
        (640, 360),
    ]


def open_auto_capture(args, state=None):
    backend_names = [args.backend] if args.backend != "auto" else ["dshow", "msmf", "any"]
    fallback = None

    for backend_name in backend_names:
        for index in range(args.auto_sources):
            if state is not None:
                state.publish_status(f"Scanning camera source={index}, backend={backend_name}...")
            backend = camera_backend(backend_name, index)
            cap = cv2.VideoCapture(index, backend)
            configure_capture(cap, args, index)
            if not cap.isOpened():
                if state is not None:
                    state.publish_status(f"Camera source={index}, backend={backend_name} did not open.")
                cap.release()
                continue

            frame = read_probe_frame(cap, attempts=args.startup_frames)
            brightness = float(frame.mean()) if frame is not None else -1.0
            if fallback is None and frame is not None:
                fallback = (index, backend, backend_name, brightness)

            if brightness >= args.black_threshold:
                print(
                    f"Auto selected camera: source={index}, backend={backend_name}, "
                    f"brightness={brightness:.1f}",
                    flush=True,
                )
                return cap, index, backend

            print(
                f"Auto skipped camera: source={index}, backend={backend_name}, "
                f"brightness={brightness:.1f}",
                flush=True,
            )
            if state is not None:
                state.publish_status(
                    f"Skipped black camera source={index}, backend={backend_name}, "
                    f"brightness={brightness:.1f}."
                )
            cap.release()

    if fallback is not None:
        index, backend, backend_name, brightness = fallback
        cap = cv2.VideoCapture(index, backend)
        configure_capture(cap, args, index)
        print(
            f"Auto fallback camera is still black: source={index}, backend={backend_name}, "
            f"brightness={brightness:.1f}",
            flush=True,
        )
        return cap, index, backend

    return cv2.VideoCapture(), "auto", cv2.CAP_ANY


def probe_cameras(args):
    original_source = args.source
    for index in range(args.probe_cameras):
        args.source = str(index)
        cap, _, _ = open_capture(args)
        if not cap.isOpened():
            print(f"source {index}: not opened")
            cap.release()
            continue

        frame = read_probe_frame(cap)
        cap.release()
        if frame is None:
            print(f"source {index}: opened, no frame")
            continue

        height, width = frame.shape[:2]
        brightness = float(frame.mean())
        status = "black" if brightness < 3 else "ok"
        print(f"source {index}: {width}x{height}, brightness={brightness:.1f}, {status}")
    args.source = original_source


def windows_camera_names():
    if platform.system().lower() != "windows":
        return []

    names = ffmpeg_dshow_camera_names()
    if names:
        return names

    queries = [
        (
            "Get-PnpDevice -PresentOnly | "
            "Where-Object { $_.FriendlyName -and ($_.Class -in @('Camera','Image') -or "
            "$_.FriendlyName -match 'Camera|Webcam|Video|OBS|NDI') } | "
            "Select-Object -ExpandProperty FriendlyName | "
            "ConvertTo-Json -Compress"
        ),
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.Name -and ($_.PNPClass -in @('Camera','Image') -or "
            "$_.Name -match 'Camera|Webcam|Video|OBS|NDI') } | "
            "Select-Object -ExpandProperty Name | "
            "ConvertTo-Json -Compress"
        ),
    ]

    for query in queries:
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            query,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        text = result.stdout.strip()
        if not text:
            continue
        try:
            names = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(names, str):
            names = [names]
        names = [name for name in names if isinstance(name, str)]
        if names:
            return names
    return []


def macos_camera_names():
    if platform.system().lower() != "darwin":
        return []
    return ffmpeg_avfoundation_camera_names()


def camera_device_names():
    system = platform.system().lower()
    if system == "windows":
        return windows_camera_names()
    if system == "darwin":
        return macos_camera_names()
    return []


def ffmpeg_dshow_camera_names():
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    text = "\n".join([result.stderr or "", result.stdout or ""])
    names = []
    for line in text.splitlines():
        match = re.search(r'\] "(.+)" \(video\)', line)
        if match:
            names.append(match.group(1))
    return names


def ffmpeg_avfoundation_camera_names():
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    text = "\n".join([result.stderr or "", result.stdout or ""])
    names = []
    in_video_section = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            names.append(match.group(2).strip())
    return names


def scan_camera_devices(args):
    devices = []
    names = camera_device_names()
    if args.backend == "auto":
        if platform.system().lower() == "windows":
            backend_names = ["dshow"]
        elif platform.system().lower() == "darwin":
            backend_names = ["avfoundation"]
        else:
            backend_names = ["any"]
    else:
        backend_names = [args.backend]
    count = max(len(names), args.auto_sources if not names else 0)

    for backend_name in backend_names:
        for index in range(count):
            devices.append(
                {
                    "source": index,
                    "backend": backend_name,
                    "name": names[index] if index < len(names) else f"Camera {index}",
                    "width": 0,
                    "height": 0,
                    "brightness": 0.0,
                    "status": "click to test",
                }
            )
    return devices


def build_records(result, frame_index, width, height, names):
    records = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return records

    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, track_id, conf, cls in zip(xyxy, ids, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        color = color_for_id(track_id)
        object_name = names.get(int(cls), str(cls))
        records.append(
            {
                "frame": frame_index,
                "track_id": int(track_id),
                "class_id": int(cls),
                "class_name": object_name,
                "confidence": round(float(conf), 4),
                "color": bgr_to_hex(color),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round(cx, 2),
                "center_y": round(cy, 2),
                "center_x_norm": round(cx / width, 6),
                "center_y_norm": round(cy / height, 6),
            }
        )
    return records


def draw_records(frame, records):
    for record in records:
        color = color_for_id(record["track_id"])
        box = [record["x1"], record["y1"], record["x2"], record["y2"]]
        draw_sticker(frame, record["track_id"], record["class_name"], box, color)


def capture_loop(args, state):
    state.set_classes(args.classes)
    state.publish_status("Starting camera capture...")
    cap, source, backend = open_capture(args, state)
    if source != "none" and not cap.isOpened():
        state.stop_event.set()
        raise RuntimeError(f"Cannot open source: {args.source}")
    print(f"Capture opened: source={source}, backend={backend}", flush=True)
    state.set_current_camera(source, backend)

    preview_frame = read_probe_frame(cap, attempts=args.startup_frames)
    if preview_frame is not None:
        height, width = preview_frame.shape[:2]
        brightness = float(preview_frame.mean())
        print(f"First frame received: {width}x{height}, brightness={brightness:.1f}", flush=True)
        if brightness < 3:
            print(
                "Warning: first camera frame is almost black. Try another --source, "
                "--backend, or check camera privacy/exposure settings.",
                flush=True,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            preview_frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                args.jpeg_quality,
                int(cv2.IMWRITE_JPEG_OPTIMIZE),
                0,
            ],
        )
        if ok:
            state.publish(
                encoded.tobytes(),
                {
                    "frame": 0,
                    "timestamp": time.time(),
                    "width": width,
                    "height": height,
                    "process_fps": 0,
                    "status": "Camera preview received. Loading YOLO model...",
                    "objects": [],
                },
            )
    else:
        print("No camera frame received during startup.", flush=True)

    model = YOLO(args.model)
    print(f"Model loaded: {args.model}", flush=True)
    frame_index = 0
    last_publish = 0.0
    started = time.perf_counter()

    try:
        while not state.stop_event.is_set():
            camera_request = state.take_camera_request()
            if camera_request is not None:
                next_source, next_backend_name = camera_request
                cap.release()
                cap, source, backend = open_requested_capture(args, next_source, next_backend_name, state)
                if not cap.isOpened():
                    state.publish_status(f"Camera source={next_source}, backend={next_backend_name} did not open.")
                    time.sleep(0.2)
                    cap, source, backend = open_capture(args, state)
                    continue
                print(f"Capture switched: source={source}, backend={backend}", flush=True)
                state.set_current_camera(source, backend)

            if not cap.isOpened():
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str) and Path(source).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if isinstance(source, int):
                    print(
                        f"Camera frame grab failed. Reopening source {source} "
                        f"with backend {args.backend}...",
                        flush=True,
                    )
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(source, backend)
                    configure_capture(cap, args, source)
                    continue
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            if args.max_fps and now - last_publish < 1.0 / args.max_fps:
                continue
            last_publish = now

            frame_index += 1
            height, width = frame.shape[:2]
            brightness = float(frame.mean())
            class_ids = state.get_class_ids()
            track_kwargs = {
                "source": frame,
                "classes": class_ids,
                "conf": args.conf,
                "device": args.device,
                "tracker": args.tracker,
                "persist": True,
                "verbose": False,
            }
            if args.imgsz:
                track_kwargs["imgsz"] = args.imgsz

            result = model.track(**track_kwargs)[0]
            records = build_records(result, frame_index, width, height, model.names)
            preview = frame.copy()
            if args.draw:
                draw_records(preview, records)

            ok, encoded = cv2.imencode(
                ".jpg",
                preview,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    args.jpeg_quality,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE),
                    0,
                ],
            )
            if not ok:
                continue

            elapsed = max(time.perf_counter() - started, 0.001)
            payload = {
                "frame": frame_index,
                "timestamp": time.time(),
                "width": width,
                "height": height,
                "process_fps": frame_index / elapsed,
                "objects": records,
            }
            if brightness < args.black_threshold:
                payload["status"] = (
                    f"Camera is returning black frames "
                    f"(brightness={brightness:.1f}). Try --source auto, another source, or another backend."
                )
            state.publish(encoded.tobytes(), payload)
    finally:
        cap.release()


def make_handler(state, args):
    class LiveHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not self.server.quiet:
                super().log_message(fmt, *args)

        def do_GET(self):
            try:
                self.route_get()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def route_get(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))
                return
            if path == "/events":
                self.stream_events()
                return
            if path == "/video":
                self.stream_video()
                return
            if path == "/latest.json":
                self.send_latest_json()
                return
            if path == "/cameras":
                self.send_cameras()
                return
            if path == "/select-camera":
                self.select_camera()
                return
            if path == "/classes":
                self.send_classes()
                return
            if path == "/set-classes":
                self.set_classes()
                return
            if path == "/favicon.ico" or path.startswith("/.well-known/"):
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def send_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_latest_json(self):
            with state.condition:
                payload = state.payload or {"frame": 0, "objects": []}
            self.send_json(payload)

        def send_cameras(self):
            cameras = scan_camera_devices(args)
            with state.condition:
                current = state.current_camera
            self.send_json({"current": current, "cameras": cameras})

        def select_camera(self):
            query = parse_qs(urlparse(self.path).query)
            source = query.get("source", ["0"])[0]
            backend = query.get("backend", ["dshow"])[0]
            state.request_camera(source, backend)
            self.send_json({"ok": True, "source": source, "backend": backend})

        def send_classes(self):
            options = [
                name
                for name in sorted(CLASS_ALIASES)
                if name != "brid"
            ]
            with state.condition:
                classes_text = state.classes_text
                class_ids = list(state.class_ids)
            self.send_json(
                {
                    "classes": classes_text,
                    "class_ids": class_ids,
                    "options": options,
                }
            )

        def set_classes(self):
            query = parse_qs(urlparse(self.path).query)
            classes_text = query.get("classes", ["person"])[0]
            try:
                normalized, class_ids = state.set_classes(classes_text)
            except ValueError as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json({"ok": True, "classes": normalized, "class_ids": class_ids})

        def stream_events(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_seq = -1
            while not state.stop_event.is_set():
                with state.condition:
                    state.condition.wait_for(lambda: state.seq != last_seq or state.stop_event.is_set(), timeout=15)
                    if state.stop_event.is_set():
                        break
                    last_seq = state.seq
                    payload = state.payload
                try:
                    if payload is None:
                        self.wfile.write(b": waiting\n\n")
                    else:
                        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break

        def stream_video(self):
            boundary = b"--frame"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_seq = -1
            while not state.stop_event.is_set():
                with state.condition:
                    state.condition.wait_for(lambda: state.seq != last_seq or state.stop_event.is_set(), timeout=15)
                    if state.stop_event.is_set():
                        break
                    last_seq = state.seq
                    jpeg = state.jpeg
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break

    return LiveHandler


def main():
    parser = argparse.ArgumentParser(description="Run live YOLO tracking from a camera or stream.")
    parser.add_argument("--source", default="none", help="Camera index, video path, stream URL, auto, or none. Default: none.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "any", "dshow", "msmf", "ffmpeg", "avfoundation"],
        help="OpenCV capture backend. Auto uses DirectShow on Windows and AVFoundation on macOS.",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path, e.g. yolo11n.pt or yolo26n.pt.")
    parser.add_argument("--classes", default="car", help="Comma-separated class IDs or aliases, e.g. person,car.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="Inference device: cpu, 0 for CUDA, or mps.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size. Smaller is faster.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    parser.add_argument("--width", type=int, default=0, help="Requested camera width. Default keeps camera output.")
    parser.add_argument("--height", type=int, default=0, help="Requested camera height. Default keeps camera output.")
    parser.add_argument(
        "--no-auto-resolution",
        dest="auto_resolution",
        action="store_false",
        help="Do not probe common HD resolutions when opening a camera.",
    )
    parser.add_argument(
        "--resolution-mode",
        default="balanced",
        choices=["speed", "balanced", "quality"],
        help="Auto camera resolution preference. Default: balanced.",
    )
    parser.add_argument("--camera-fps", type=float, default=0, help="Requested camera FPS.")
    parser.add_argument("--max-fps", type=float, default=0, help="Limit processed FPS. 0 means unlimited.")
    parser.add_argument("--jpeg-quality", type=int, default=96, help="MJPEG JPEG quality, 1-100.")
    parser.add_argument("--draw", action="store_true", help="Also draw server-side stickers into the MJPEG stream.")
    parser.add_argument("--quiet", action="store_true", help="Hide HTTP access logs.")
    parser.add_argument("--auto-sources", type=int, default=6, help="Number of camera indexes to scan for --source auto.")
    parser.add_argument("--startup-frames", type=int, default=45, help="Frames to sample during camera startup/probing.")
    parser.add_argument("--black-threshold", type=float, default=3.0, help="Brightness below this is treated as black.")
    parser.add_argument(
        "--probe-cameras",
        type=int,
        default=0,
        help="Probe camera indexes from 0 to N-1 and exit.",
    )
    parser.set_defaults(auto_resolution=True)
    args = parser.parse_args()

    if args.probe_cameras:
        probe_cameras(args)
        return

    state = LiveState()
    worker = threading.Thread(target=capture_loop, args=(args, state), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args))
    server.quiet = args.quiet
    print(f"Live tracking: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        with state.condition:
            state.condition.notify_all()
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()

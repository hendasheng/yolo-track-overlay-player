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
import numpy as np
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
      #browserPreview, #overlay {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      #browserPreview { background: #050706; }
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
      .switch-field {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        min-height: 34px;
      }
      .switch-field input {
        width: 18px;
        height: 18px;
        accent-color: #88c7ff;
      }
      @media (max-width: 700px) {
        .hud {
          left: 10px;
          right: 10px;
          top: auto;
          bottom: 10px;
          grid-template-columns: repeat(5, 1fr);
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
      <video id="browserPreview" autoplay muted playsinline></video>
      <canvas id="overlay"></canvas>
    </main>
    <section class="control-panel">
      <header>
        <strong>Live Controls</strong>
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
      <label class="field switch-field">
        <span>Detection</span>
        <input id="detectionToggle" type="checkbox" />
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
      <div><span>Detect</span><strong id="detectStat">-</strong></div>
      <div><span>Size</span><strong id="sizeStat">-</strong></div>
    </aside>
    <script>
      const browserPreview = document.querySelector("#browserPreview");
      const canvas = document.querySelector("#overlay");
      const ctx = canvas.getContext("2d");
      const frameStat = document.querySelector("#frameStat");
      const objectStat = document.querySelector("#objectStat");
      const fpsStat = document.querySelector("#fpsStat");
      const detectStat = document.querySelector("#detectStat");
      const sizeStat = document.querySelector("#sizeStat");
      const status = document.querySelector("#status");
      const cameraSelect = document.querySelector("#cameraSelect");
      const cameraDetail = document.querySelector("#cameraDetail");
      const classSelect = document.querySelector("#classSelect");
      const detectionToggle = document.querySelector("#detectionToggle");
      const customClassWrap = document.querySelector("#customClassWrap");
      const customClassInput = document.querySelector("#customClassInput");
      let latest = { width: 1, height: 1, objects: [] };
      let streamNaturalWidth = 1;
      let streamNaturalHeight = 1;
      let pendingCameraValue = "";
      let currentCameraValue = "";
      let browserStream = null;
      let detectionTimer = 0;
      let detecting = false;
      let detectFrame = 0;
      let detectStarted = 0;
      let latestDetectionAt = 0;
      let configPromise = null;
      const detectCanvas = document.createElement("canvas");
      const detectCtx = detectCanvas.getContext("2d");

      async function getConfig() {
        if (!configPromise) {
          configPromise = fetch("/config").then((response) => response.json()).catch(() => ({ detect_width: 480 }));
        }
        return configPromise;
      }

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
        if (latest.viewport_space) {
          const age = latestDetectionAt ? performance.now() - latestDetectionAt : 0;
          if (age <= 220) {
            for (const obj of latest.objects || []) drawViewportObject(obj, size.w, size.h);
          }
          requestAnimationFrame(draw);
          return;
        }
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

      function drawViewportObject(obj, viewportW, viewportH) {
        const x = obj.x1;
        const y = obj.y1;
        const w = obj.x2 - obj.x1;
        const h = obj.y2 - obj.y1;
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
        const labelX = Math.max(0, Math.min(x, viewportW - labelW));
        const labelY = Math.max(0, y - labelH);
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY, labelW, labelH);
        ctx.fillStyle = "#050706";
        ctx.textBaseline = "middle";
        ctx.fillText(label, labelX + 7, labelY + labelH / 2);
      }

      browserPreview.onloadedmetadata = () => {
        streamNaturalWidth = browserPreview.videoWidth || streamNaturalWidth;
        streamNaturalHeight = browserPreview.videoHeight || streamNaturalHeight;
        latest.width = streamNaturalWidth;
        latest.height = streamNaturalHeight;
        sizeStat.textContent = `${streamNaturalWidth}x${streamNaturalHeight}`;
      };
      window.addEventListener("resize", () => {
        resizeCanvas();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (latest.viewport_space) latest.objects = [];
      });
      async function refreshCameras() {
        const previousValue = pendingCameraValue || currentCameraValue || cameraSelect.value;
        cameraSelect.innerHTML = "";
        const loading = document.createElement("option");
        loading.textContent = "Scanning...";
        loading.value = "";
        cameraSelect.appendChild(loading);
        cameraDetail.textContent = "Scanning camera devices...";
        try {
          let devices = await navigator.mediaDevices.enumerateDevices();
          if (!devices.some((device) => device.kind === "videoinput" && device.label)) {
            const permissionStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            for (const track of permissionStream.getTracks()) track.stop();
            devices = await navigator.mediaDevices.enumerateDevices();
          }
          const cameras = devices.filter((device) => device.kind === "videoinput");
          cameraSelect.innerHTML = "";
          const placeholder = document.createElement("option");
          placeholder.textContent = "Choose camera";
          placeholder.value = "";
          cameraSelect.appendChild(placeholder);
          for (const camera of cameras) {
            const option = document.createElement("option");
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${cameraSelect.options.length}`;
            option.dataset.detail = option.textContent;
            cameraSelect.appendChild(option);
          }
          const targetValue = previousValue || currentCameraValue;
          if (targetValue) {
            const match = [...cameraSelect.options].find((option) => option.value === targetValue);
            if (match) {
              cameraSelect.value = targetValue;
              cameraDetail.textContent = match.dataset.detail || "";
            }
          }
          if (!cameraSelect.value) {
            if (cameraSelect.options.length === 1) cameraDetail.textContent = "No camera found.";
            else cameraDetail.textContent = "Choose a camera.";
          }
        } catch (error) {
          cameraSelect.innerHTML = "";
          cameraDetail.textContent = "Camera scan failed.";
        }
      }
      cameraSelect.onchange = async () => {
        const deviceId = cameraSelect.value;
        const selected = cameraSelect.selectedOptions[0];
        pendingCameraValue = cameraSelect.value;
        cameraDetail.textContent = selected?.dataset.detail || "";
        status.textContent = `Switching to ${selected?.textContent || "camera"}...`;
        status.classList.remove("is-hidden");
        await startBrowserCamera(deviceId);
        currentCameraValue = deviceId;
      };
      function stopBrowserCamera() {
        if (browserStream) {
          for (const track of browserStream.getTracks()) track.stop();
          browserStream = null;
        }
        browserPreview.srcObject = null;
      }
      async function startBrowserCamera(deviceId) {
        stopBrowserCamera();
        const constraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId } }
            : { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        browserStream = await navigator.mediaDevices.getUserMedia(constraints);
        browserPreview.srcObject = browserStream;
        await browserPreview.play();
        status.classList.add("is-hidden");
      }
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
      detectionToggle.onchange = async () => {
        if (detectionToggle.checked) startDetectionLoop();
        else stopDetectionLoop();
      };
      function startDetectionLoop() {
        if (detectionTimer) return;
        detectStarted = performance.now();
        latestDetectionAt = 0;
        status.textContent = "Detection on";
        status.classList.remove("is-hidden");
        detectionTimer = window.setInterval(sendDetectionFrame, 80);
      }
      function stopDetectionLoop() {
        window.clearInterval(detectionTimer);
        detectionTimer = 0;
        detecting = false;
        latest.objects = [];
        latestDetectionAt = 0;
        objectStat.textContent = "0";
        fpsStat.textContent = "-";
        detectStat.textContent = "-";
        status.textContent = "Detection off";
        status.classList.remove("is-hidden");
      }
      async function sendDetectionFrame() {
        if (detecting || !browserPreview.videoWidth || browserPreview.readyState < 2) return;
        detecting = true;
        try {
          const viewport = canvas.getBoundingClientRect();
          const viewportWidth = Math.max(1, Math.floor(viewport.width));
          const viewportHeight = Math.max(1, Math.floor(viewport.height));
          const videoWidth = browserPreview.videoWidth;
          const videoHeight = browserPreview.videoHeight;
          const viewportRatio = viewportWidth / viewportHeight;
          const videoRatio = videoWidth / videoHeight;
          let sx = 0;
          let sy = 0;
          let sw = videoWidth;
          let sh = videoHeight;
          if (viewportRatio > videoRatio) {
            sh = videoWidth / viewportRatio;
            sy = (videoHeight - sh) / 2;
          } else {
            sw = videoHeight * viewportRatio;
            sx = (videoWidth - sw) / 2;
          }
          const config = await getConfig();
          const detectWidth = Math.min(config.detect_width || 480, viewportWidth);
          const detectHeight = Math.max(1, Math.round(viewportHeight * (detectWidth / viewportWidth)));
          detectCanvas.width = detectWidth;
          detectCanvas.height = detectHeight;
          detectCtx.drawImage(browserPreview, sx, sy, sw, sh, 0, 0, detectWidth, detectHeight);
          const blob = await new Promise((resolve) => detectCanvas.toBlob(resolve, "image/jpeg", 0.8));
          if (!blob) return;
          const response = await fetch(`/detect-frame?frame=${++detectFrame}&source_width=${viewportWidth}&source_height=${viewportHeight}`, {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: blob,
          });
          const data = await response.json();
          latest = data;
          latestDetectionAt = performance.now();
          frameStat.textContent = data.frame || "-";
          objectStat.textContent = (data.objects || []).length;
          const elapsed = Math.max((performance.now() - detectStarted) / 1000, 0.001);
          fpsStat.textContent = (detectFrame / elapsed).toFixed(1);
          sizeStat.textContent = data.width && data.height ? `${data.width}x${data.height}` : "-";
          detectStat.textContent = data.detect_ms ? `${data.detect_ms}ms` : "-";
          status.classList.add("is-hidden");
        } catch (error) {
          status.textContent = "Detection request failed.";
          status.classList.remove("is-hidden");
        } finally {
          detecting = false;
        }
      }
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
        self.detection_enabled = False
        self.video_clients = 0
        self.detect_condition = threading.Condition()
        self.detect_frame = None
        self.detect_frame_index = 0
        self.detect_size = (0, 0)
        self.detect_seq = 0
        self.latest_records = []
        self.inference_count = 0
        self.http_model = None
        self.http_model_lock = threading.Lock()

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

    def clear_current_camera(self):
        with self.condition:
            self.current_camera = None

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

    def set_detection_enabled(self, enabled):
        with self.condition:
            self.detection_enabled = bool(enabled)
            self.payload = {
                "frame": 0,
                "timestamp": time.time(),
                "width": 1,
                "height": 1,
                "process_fps": 0,
                "status": "Detection on" if enabled else "Detection off",
                "objects": [],
            }
            self.seq += 1
            self.condition.notify_all()

    def is_detection_enabled(self):
        with self.condition:
            return self.detection_enabled

    def add_video_client(self):
        with self.condition:
            self.video_clients += 1

    def remove_video_client(self):
        with self.condition:
            self.video_clients = max(0, self.video_clients - 1)

    def has_video_clients(self):
        with self.condition:
            return self.video_clients > 0

    def submit_detection_frame(self, frame, frame_index, width, height):
        with self.detect_condition:
            self.detect_frame = frame
            self.detect_frame_index = frame_index
            self.detect_size = (width, height)
            self.detect_seq += 1
            self.detect_condition.notify()

    def wait_detection_frame(self, last_seq, timeout=0.2):
        with self.detect_condition:
            self.detect_condition.wait_for(
                lambda: self.detect_seq != last_seq or self.stop_event.is_set(),
                timeout=timeout,
            )
            if self.stop_event.is_set() or self.detect_seq == last_seq or self.detect_frame is None:
                return None
            return (
                self.detect_seq,
                self.detect_frame,
                self.detect_frame_index,
                self.detect_size,
            )

    def set_latest_records(self, records):
        with self.condition:
            self.latest_records = records
            self.inference_count += 1

    def get_latest_records(self):
        with self.condition:
            return list(self.latest_records), self.inference_count

    def clear_latest_records(self):
        with self.condition:
            self.latest_records = []


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


def open_video_capture(source, backend_name):
    backend = camera_backend(backend_name, source)
    system = platform.system().lower()
    attempts = [(backend_name, lambda: cv2.VideoCapture(source, backend))]

    # On macOS, some OpenCV builds behave better if we let the backend auto-resolve
    # after an explicit AVFoundation attempt.
    if system == "darwin" and isinstance(source, int) and backend_name in {"auto", "any", "avfoundation"}:
        attempts.append(("any", lambda: cv2.VideoCapture(source)))

    last_cap = None
    for label, opener in attempts:
        cap = opener()
        last_cap = cap
        if cap.isOpened():
            return cap, backend if label == backend_name else cv2.CAP_ANY, label
        cap.release()

    return last_cap or cv2.VideoCapture(), backend, backend_name


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


def open_capture(args, state=None, fast=False):
    source = parse_source(args.source)
    if source == "none":
        if state is not None:
            state.publish_status("Select a camera from the Cameras panel.")
        return cv2.VideoCapture(), "none", cv2.CAP_ANY, "any"
    if source == "auto":
        return open_auto_capture(args, state, fast=fast)

    if state is not None:
        state.publish_status(f"Opening camera source={source}, backend={args.backend}...")
    cap, backend, opened_backend_name = open_video_capture(source, args.backend)
    configure_capture(cap, args, source, fast=fast)
    if state is not None and opened_backend_name != args.backend and cap.isOpened():
        state.publish_status(
            f"Opening camera source={source}, backend={args.backend}... fallback to {opened_backend_name}."
        )
    return cap, source, backend, opened_backend_name


def open_requested_capture(args, source, backend_name, state=None, fast=False):
    previous_source = args.source
    previous_backend = args.backend
    args.source = str(source)
    args.backend = backend_name
    try:
        return open_capture(args, state, fast=fast)
    finally:
        args.source = previous_source
        args.backend = previous_backend


def configure_capture(cap, args, source=None, fast=False):
    if args.width or args.height:
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    elif args.auto_resolution and isinstance(source, int):
        width, height = resolution_request(args.resolution_mode)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if args.camera_fps:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)


def resolution_request(mode):
    if mode == "quality":
        return 1920, 1080
    if mode == "speed":
        return 960, 540
    return 1280, 720


def open_auto_capture(args, state=None, fast=False):
    if args.backend != "auto":
        backend_names = [args.backend]
    elif platform.system().lower() == "darwin":
        backend_names = ["avfoundation", "any"]
    else:
        backend_names = ["dshow", "msmf", "any"]
    fallback = None

    for backend_name in backend_names:
        for index in range(args.auto_sources):
            if state is not None:
                state.publish_status(f"Scanning camera source={index}, backend={backend_name}...")
            cap, backend, opened_backend_name = open_video_capture(index, backend_name)
            configure_capture(cap, args, index, fast=fast)
            if not cap.isOpened():
                if state is not None:
                    state.publish_status(f"Camera source={index}, backend={backend_name} did not open.")
                cap.release()
                continue

            frame = read_probe_frame(cap, attempts=args.startup_frames)
            brightness = float(frame.mean()) if frame is not None else -1.0
            if fallback is None and frame is not None:
                fallback = (index, backend, opened_backend_name, brightness)

            if brightness >= args.black_threshold:
                print(
                    f"Auto selected camera: source={index}, backend={opened_backend_name}, "
                    f"brightness={brightness:.1f}",
                    flush=True,
                )
                return cap, index, backend, opened_backend_name

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
        configure_capture(cap, args, index, fast=fast)
        print(
            f"Auto fallback camera is still black: source={index}, backend={backend_name}, "
            f"brightness={brightness:.1f}",
            flush=True,
        )
        return cap, index, backend, backend_name

    return cv2.VideoCapture(), "auto", cv2.CAP_ANY, "any"


def probe_cameras(args):
    original_source = args.source
    for index in range(args.probe_cameras):
        args.source = str(index)
        cap, _, _, _ = open_capture(args)
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


def ffmpeg_avfoundation_camera_devices():
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
    devices = []
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
            devices.append({"source": int(match.group(1)), "name": match.group(2).strip()})
    return devices


def camera_device_entries():
    system = platform.system().lower()
    if system == "windows":
        return [{"source": index, "name": name} for index, name in enumerate(ffmpeg_dshow_camera_names())]
    if system == "darwin":
        return ffmpeg_avfoundation_camera_devices()
    return []


def scan_camera_devices(args, current=None):
    devices = []
    detected_entries = camera_device_entries()
    detected_by_source = {entry["source"]: entry["name"] for entry in detected_entries}
    if args.backend == "auto":
        if platform.system().lower() == "windows":
            backend_names = ["dshow"]
        elif platform.system().lower() == "darwin":
            backend_names = ["avfoundation"]
        else:
            backend_names = ["any"]
    else:
        backend_names = [args.backend]
    current_source = -1
    if current and isinstance(current.get("source"), int):
        current_source = current["source"]

    for backend_name in backend_names:
        if backend_name == "avfoundation" and detected_entries:
            source_indexes = sorted(set(detected_by_source.keys()) | ({current_source} if current_source >= 0 else set()))
        else:
            count = max(args.auto_sources, current_source + 1, len(detected_entries))
            source_indexes = list(range(count))

        for index in source_indexes:
            devices.append(
                {
                    "source": index,
                    "backend": backend_name,
                    "name": detected_by_source.get(index, f"Camera {index}"),
                    "width": 0,
                    "height": 0,
                    "brightness": 0.0,
                    "status": "click to open",
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


def scale_records(records, source_width, source_height, detect_width, detect_height):
    if source_width == detect_width and source_height == detect_height:
        return records

    scale_x = source_width / detect_width
    scale_y = source_height / detect_height
    scaled = []
    for record in records:
        next_record = dict(record)
        next_record["x1"] = round(record["x1"] * scale_x, 2)
        next_record["y1"] = round(record["y1"] * scale_y, 2)
        next_record["x2"] = round(record["x2"] * scale_x, 2)
        next_record["y2"] = round(record["y2"] * scale_y, 2)
        next_record["center_x"] = round(record["center_x"] * scale_x, 2)
        next_record["center_y"] = round(record["center_y"] * scale_y, 2)
        next_record["center_x_norm"] = round(next_record["center_x"] / source_width, 6)
        next_record["center_y_norm"] = round(next_record["center_y"] / source_height, 6)
        scaled.append(next_record)
    return scaled


def draw_records(frame, records):
    for record in records:
        color = color_for_id(record["track_id"])
        box = [record["x1"], record["y1"], record["x2"], record["y2"]]
        draw_sticker(frame, record["track_id"], record["class_name"], box, color)


def publish_preview_frame(state, cap, args, status_text):
    preview_frame = read_probe_frame(cap, attempts=min(args.startup_frames, 10), delay=0.02)
    if preview_frame is None:
        return False

    height, width = preview_frame.shape[:2]
    brightness = float(preview_frame.mean())
    print(f"First frame received: {width}x{height}, brightness={brightness:.1f}", flush=True)
    if brightness < 3:
        print(
            "Warning: first camera frame is almost black. Try another --source, "
            "--backend, or check camera privacy/exposure settings.",
            flush=True,
        )
    encoded_frame = resize_stream_frame(preview_frame, args.stream_width)
    stream_height, stream_width = encoded_frame.shape[:2]
    ok, encoded = cv2.imencode(
        ".jpg",
        encoded_frame,
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
                "source_width": width,
                "source_height": height,
                "stream_width": stream_width,
                "stream_height": stream_height,
                "process_fps": 0,
                "status": status_text,
                "objects": [],
            },
        )
    return True


def detection_loop(args, state):
    model = None
    last_seq = 0
    last_inference = 0.0

    while not state.stop_event.is_set():
        item = state.wait_detection_frame(last_seq)
        if item is None:
            continue
        seq, frame, frame_index, size = item
        last_seq = seq

        if not state.is_detection_enabled():
            state.clear_latest_records()
            continue

        now = time.perf_counter()
        if args.max_fps and now - last_inference < 1.0 / args.max_fps:
            continue

        if model is None:
            state.publish_status("Loading YOLO model...")
            model = YOLO(args.model)
            print(f"Model loaded: {args.model}", flush=True)

        width, height = size
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
        state.set_latest_records(records)
        last_inference = now


def capture_loop(args, state):
    state.set_classes(args.classes)
    source = parse_source(args.source)
    backend = cv2.CAP_ANY
    backend_name = "any"
    cap = cv2.VideoCapture()
    if source == "none":
        state.clear_current_camera()
        state.publish_status("Select a camera from the Cameras panel.")
    else:
        state.publish_status("Starting camera capture...")
        cap, source, backend, backend_name = open_capture(args, state)
        if not cap.isOpened():
            state.stop_event.set()
            raise RuntimeError(f"Cannot open source: {args.source}")
        print(f"Capture opened: source={source}, backend={backend_name}", flush=True)
        state.set_current_camera(source, backend_name)
        if not publish_preview_frame(state, cap, args, "Camera preview received. Loading YOLO model..."):
            print("No camera frame received during startup.", flush=True)

    frame_index = 0
    last_stream = 0.0
    last_detection_submit = 0.0
    started = time.perf_counter()
    preview_interval = 1.0 / max(args.stream_fps, 1.0)
    detection_interval = 1.0 / max(args.max_fps, 1.0) if args.max_fps else 0.0

    try:
        while not state.stop_event.is_set():
            camera_request = state.take_camera_request()
            if camera_request is not None:
                next_source, next_backend_name = camera_request
                if cap.isOpened():
                    cap.release()
                if str(next_source).lower() in {"none", "off"}:
                    source = "none"
                    backend = cv2.CAP_ANY
                    backend_name = "any"
                    cap = cv2.VideoCapture()
                    state.clear_current_camera()
                    state.publish_status("Select a camera from the Cameras panel.")
                    continue
                cap, source, backend, backend_name = open_requested_capture(
                    args,
                    next_source,
                    next_backend_name,
                    state,
                    fast=True,
                )
                if not cap.isOpened():
                    source = "none"
                    backend = cv2.CAP_ANY
                    backend_name = "any"
                    state.clear_current_camera()
                    state.publish_status(f"Camera source={next_source}, backend={next_backend_name} did not open.")
                    continue
                print(f"Capture switched: source={source}, backend={backend_name}", flush=True)
                state.set_current_camera(source, backend_name)
                if not publish_preview_frame(state, cap, args, f"Camera switched: source={source}, backend={backend_name}"):
                    state.publish_status(f"Camera source={source}, backend={backend_name} opened, waiting for frames...")

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
                        f"with backend {backend_name}...",
                        flush=True,
                    )
                    cap.release()
                    time.sleep(0.5)
                    cap, backend, backend_name = open_video_capture(source, backend_name)
                    configure_capture(cap, args, source, fast=True)
                    continue
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            frame_index += 1
            height, width = frame.shape[:2]
            brightness = float(frame.mean())
            detection_enabled = state.is_detection_enabled()
            if not detection_enabled:
                state.clear_latest_records()
            else:
                if not detection_interval or now - last_detection_submit >= detection_interval:
                    state.submit_detection_frame(frame, frame_index, width, height)
                    last_detection_submit = now
            records, inference_count = state.get_latest_records()

            if now - last_stream < preview_interval:
                continue
            last_stream = now

            if not state.has_video_clients():
                continue

            preview = frame.copy()
            if args.draw:
                draw_records(preview, records)
            preview = resize_stream_frame(preview, args.stream_width)
            stream_height, stream_width = preview.shape[:2]

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
                "source_width": width,
                "source_height": height,
                "stream_width": stream_width,
                "stream_height": stream_height,
                "process_fps": inference_count / elapsed,
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


def resize_stream_frame(frame, stream_width):
    if not stream_width or stream_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= stream_width:
        return frame
    ratio = stream_width / width
    stream_height = max(1, int(height * ratio))
    return cv2.resize(frame, (stream_width, stream_height), interpolation=cv2.INTER_AREA)


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

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                if path == "/detect-frame":
                    self.detect_frame()
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
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
            if path == "/config":
                self.send_config()
                return
            if path == "/set-classes":
                self.set_classes()
                return
            if path == "/set-detection":
                self.set_detection()
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
            with state.condition:
                current = state.current_camera
            cameras = scan_camera_devices(args, current)
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

        def set_detection(self):
            query = parse_qs(urlparse(self.path).query)
            value = query.get("enabled", ["0"])[0].strip().lower()
            enabled = value in {"1", "true", "yes", "on"}
            state.set_detection_enabled(enabled)
            self.send_json({"ok": True, "enabled": enabled})

        def detect_frame(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            raw = self.rfile.read(length)
            frame_data = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            query = parse_qs(urlparse(self.path).query)
            frame_index = int(query.get("frame", ["0"])[0] or 0)
            detect_height, detect_width = frame.shape[:2]
            source_width = int(query.get("source_width", [str(detect_width)])[0] or detect_width)
            source_height = int(query.get("source_height", [str(detect_height)])[0] or detect_height)

            started = time.perf_counter()
            with state.http_model_lock:
                if state.http_model is None:
                    state.http_model = YOLO(args.model)
                    print(f"Model loaded: {args.model}", flush=True)
                model = state.http_model
                track_kwargs = {
                    "source": frame,
                    "classes": state.get_class_ids(),
                    "conf": args.conf,
                    "device": args.device,
                    "tracker": args.tracker,
                    "persist": True,
                    "verbose": False,
                }
                if args.imgsz:
                    track_kwargs["imgsz"] = args.imgsz
                result = model.track(**track_kwargs)[0]
                records = build_records(result, frame_index, detect_width, detect_height, model.names)
                records = scale_records(records, source_width, source_height, detect_width, detect_height)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

            self.send_json(
                {
                    "frame": frame_index,
                    "timestamp": time.time(),
                    "width": source_width,
                    "height": source_height,
                    "source_width": source_width,
                    "source_height": source_height,
                    "detect_width": detect_width,
                    "detect_height": detect_height,
                    "detect_ms": elapsed_ms,
                    "viewport_space": True,
                    "objects": records,
                }
            )

        def send_config(self):
            self.send_json(
                {
                    "detect_width": args.detect_width,
                    "max_fps": args.max_fps,
                    "imgsz": args.imgsz,
                }
            )

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
            state.add_video_client()
            try:
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
            finally:
                state.remove_video_client()

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
        help="Do not request a resolution preset when opening a camera.",
    )
    parser.add_argument(
        "--resolution-mode",
        default="balanced",
        choices=["speed", "balanced", "quality"],
        help="Auto camera resolution preference. Default: balanced.",
    )
    parser.add_argument("--camera-fps", type=float, default=0, help="Requested camera FPS.")
    parser.add_argument("--max-fps", type=float, default=12, help="Limit processed FPS. Default: 12.")
    parser.add_argument("--detect-width", type=int, default=480, help="Browser frame width sent to detector. Default: 480.")
    parser.add_argument("--stream-width", type=int, default=1280, help="Max MJPEG stream width. Use 0 for source width.")
    parser.add_argument("--stream-fps", type=float, default=24, help="MJPEG stream FPS. Default: 24.")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="MJPEG JPEG quality, 1-100. Default: 75.")
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
    capture_worker = threading.Thread(target=capture_loop, args=(args, state), daemon=True)
    detect_worker = threading.Thread(target=detection_loop, args=(args, state), daemon=True)
    capture_worker.start()
    detect_worker.start()

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
        with state.detect_condition:
            state.detect_condition.notify_all()
        capture_worker.join(timeout=3)
        detect_worker.join(timeout=3)


if __name__ == "__main__":
    main()

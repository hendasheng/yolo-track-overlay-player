const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");

const dropLayer = document.querySelector("#dropLayer");
const panel = document.querySelector(".panel");
const videoInput = document.querySelector("#videoInput");
const tracksInput = document.querySelector("#tracksInput");
const playToggle = document.querySelector("#playToggle");
const videoName = document.querySelector("#videoName");
const tracksName = document.querySelector("#tracksName");
const fpsInput = document.querySelector("#fpsInput");
const opacityInput = document.querySelector("#opacityInput");
const modeInput = document.querySelector("#modeInput");
const frameStat = document.querySelector("#frameStat");
const objectStat = document.querySelector("#objectStat");
const trackStat = document.querySelector("#trackStat");

let tracksByFrame = new Map();
let maxFrame = 0;
let rafId = 0;
let videoFrameCallbackId = 0;
let canvasCssWidth = 0;
let canvasCssHeight = 0;
let cachedFit = null;
let lastRenderedFrame = 0;
let lastRenderedTime = -1;
let lastStatsFrame = 0;
let lastPlaybackTime = 0;
let panelHideTimer = 0;

const panelHotspotSize = 150;
const compactPanelMedia = window.matchMedia("(max-width: 860px)");

video.loop = true;

function parseTrackText(text, filename) {
  const trimmed = text.trim();
  if (!trimmed) return new Map();

  const map = new Map();
  const isJsonl = filename.toLowerCase().endsWith(".jsonl") || trimmed.includes("\n{");
  const frames = isJsonl
    ? trimmed.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))
    : JSON.parse(trimmed);

  const list = Array.isArray(frames) ? frames : Object.values(frames);
  maxFrame = 0;

  for (const row of list) {
    const frame = Number(row.frame);
    const objects = Array.isArray(row.objects) ? row.objects : [];
    if (!Number.isFinite(frame)) continue;
    map.set(frame, objects);
    maxFrame = Math.max(maxFrame, frame);
  }

  return map;
}

function colorForTrack(id, alpha = 1) {
  const n = Number(id || 0) * 47;
  const hue = (n * 29) % 360;
  return `hsla(${hue}, 90%, 60%, ${alpha})`;
}

function colorWithAlpha(color, alpha = 1) {
  if (typeof color !== "string") return "";

  const match = color.trim().match(/^#?([0-9a-f]{6})$/i);
  if (!match) return "";

  const hex = match[1];
  const red = Number.parseInt(hex.slice(0, 2), 16);
  const green = Number.parseInt(hex.slice(2, 4), 16);
  const blue = Number.parseInt(hex.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function colorForObject(obj, alpha = 1) {
  return colorWithAlpha(obj.color, alpha) || colorForTrack(obj.track_id, alpha);
}

function computeFitRect() {
  const bounds = canvas.getBoundingClientRect();
  const videoRatio = video.videoWidth / video.videoHeight;
  const boundsRatio = bounds.width / bounds.height;

  if (!video.videoWidth || !video.videoHeight) {
    return { x: 0, y: 0, width: bounds.width, height: bounds.height, scaleX: 1, scaleY: 1 };
  }

  let width;
  let height;
  let x;
  let y;

  if (boundsRatio > videoRatio) {
    width = bounds.width;
    height = width / videoRatio;
    x = 0;
    y = (bounds.height - height) / 2;
  } else {
    height = bounds.height;
    width = height * videoRatio;
    x = (bounds.width - width) / 2;
    y = 0;
  }

  return {
    x,
    y,
    width,
    height,
    scaleX: width / video.videoWidth,
    scaleY: height / video.videoHeight,
  };
}

function getFitRect() {
  if (!cachedFit) cachedFit = computeFitRect();
  return cachedFit;
}

function resizeCanvas() {
  const bounds = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const nextCssWidth = Math.max(1, Math.floor(bounds.width));
  const nextCssHeight = Math.max(1, Math.floor(bounds.height));
  const nextWidth = Math.max(1, Math.floor(nextCssWidth * dpr));
  const nextHeight = Math.max(1, Math.floor(nextCssHeight * dpr));

  if (
    canvas.width === nextWidth &&
    canvas.height === nextHeight &&
    canvasCssWidth === nextCssWidth &&
    canvasCssHeight === nextCssHeight
  ) {
    return;
  }

  canvasCssWidth = nextCssWidth;
  canvasCssHeight = nextCssHeight;
  canvas.width = nextWidth;
  canvas.height = nextHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cachedFit = null;
}

function inferFps() {
  if (maxFrame && video.duration && Number.isFinite(video.duration)) {
    const fps = maxFrame / video.duration;
    fpsInput.value = fps.toFixed(2);
  }
}

function invalidateRender() {
  lastRenderedFrame = 0;
  lastRenderedTime = -1;
  lastStatsFrame = 0;
  cachedFit = null;
}

function setPanelOpen(isOpen) {
  panel.classList.toggle("is-open", isOpen);
}

function isPointInPanel(clientX, clientY) {
  const rect = panel.getBoundingClientRect();
  return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
}

function isPointInPanelHotspot(clientX, clientY) {
  if (compactPanelMedia.matches) {
    return clientY >= window.innerHeight - panelHotspotSize;
  }

  return clientX >= window.innerWidth - panelHotspotSize && clientY <= panelHotspotSize;
}

function schedulePanelClose() {
  window.clearTimeout(panelHideTimer);
  panelHideTimer = window.setTimeout(() => setPanelOpen(false), 140);
}

function updatePanelFromPointer(event) {
  window.clearTimeout(panelHideTimer);

  if (isPointInPanelHotspot(event.clientX, event.clientY) || isPointInPanel(event.clientX, event.clientY)) {
    setPanelOpen(true);
  } else {
    schedulePanelClose();
  }
}

function drawObject(obj, fit, opacity, mode) {
  const x = fit.x + obj.x1 * fit.scaleX;
  const y = fit.y + obj.y1 * fit.scaleY;
  const w = (obj.x2 - obj.x1) * fit.scaleX;
  const h = (obj.y2 - obj.y1) * fit.scaleY;
  const cx = fit.x + obj.center_x * fit.scaleX;
  const cy = fit.y + obj.center_y * fit.scaleY;
  const fillColor = colorForObject(obj, opacity);
  const solidColor = colorForObject(obj, 0.95);

  ctx.fillStyle = fillColor;
  ctx.strokeStyle = solidColor;
  ctx.lineWidth = 2;

  if (mode === "ellipse") {
    ctx.beginPath();
    ctx.ellipse(cx, cy, Math.max(4, w / 2), Math.max(4, h / 2), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else if (mode === "center") {
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(8, Math.min(w, h) * 0.2), 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }

  const label = `${obj.class_name || "obj"} ${obj.track_id}`;
  const labelHeight = 24;
  const labelPaddingX = 8;
  ctx.font = "700 12px Segoe UI, sans-serif";
  const labelWidth = Math.ceil(ctx.measureText(label).width) + labelPaddingX * 2;
  const labelX = Math.max(0, Math.min(cx - labelWidth / 2, canvasCssWidth - labelWidth));
  const labelY = Math.max(0, y - labelHeight);

  ctx.fillStyle = solidColor;
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
  ctx.fillStyle = "#070807";
  ctx.textBaseline = "middle";
  ctx.fillText(label, labelX + labelPaddingX, labelY + labelHeight / 2);
  ctx.textBaseline = "alphabetic";
}

function render() {
  resizeCanvas();

  const fps = Number(fpsInput.value) || 30;
  const frame = Math.max(1, Math.round(video.currentTime * fps) + 1);
  const shouldRedraw = frame !== lastRenderedFrame || video.currentTime !== lastRenderedTime || video.paused;
  if (!shouldRedraw) return;

  ctx.clearRect(0, 0, canvasCssWidth, canvasCssHeight);

  const objects = tracksByFrame.get(frame) || [];
  const fit = getFitRect();
  const opacity = Number(opacityInput.value);
  const mode = modeInput.value;

  for (const obj of objects) drawObject(obj, fit, opacity, mode);

  if (frame !== lastStatsFrame) {
    frameStat.textContent = String(frame);
    objectStat.textContent = String(objects.length);
    trackStat.textContent = String(new Set(objects.map((obj) => obj.track_id)).size);
    lastStatsFrame = frame;
  }

  lastRenderedFrame = frame;
  lastRenderedTime = video.currentTime;
}

function renderLoop() {
  render();
  rafId = requestAnimationFrame(renderLoop);
}

function renderVideoFrame() {
  render();
  if (!video.paused && !video.ended) {
    videoFrameCallbackId = video.requestVideoFrameCallback(renderVideoFrame);
  }
}

function startRendering() {
  stopRendering();
  if ("requestVideoFrameCallback" in video) {
    videoFrameCallbackId = video.requestVideoFrameCallback(renderVideoFrame);
  } else {
    rafId = requestAnimationFrame(renderLoop);
  }
}

function stopRendering() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = 0;
  }
  if (videoFrameCallbackId && "cancelVideoFrameCallback" in video) {
    video.cancelVideoFrameCallback(videoFrameCallbackId);
    videoFrameCallbackId = 0;
  }
}

function waitForVideoMetadata() {
  if (!video.src || video.readyState >= 1) return Promise.resolve();

  return new Promise((resolve) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

async function playFromStart() {
  if (!video.src) {
    invalidateRender();
    render();
    return;
  }

  stopRendering();
  await waitForVideoMetadata();
  video.currentTime = 0;
  invalidateRender();
  render();

  await video
    .play()
    .then(() => {
      startRendering();
    })
    .catch((error) => {
      console.error("Video playback failed:", error);
      playToggle.textContent = "Play";
    });
}

async function loadVideo(file) {
  if (video.src) URL.revokeObjectURL(video.src);
  stopRendering();
  video.src = URL.createObjectURL(file);
  videoName.textContent = file.name;
  dropLayer.classList.add("is-hidden");
  invalidateRender();
  render();
}

async function loadTracks(file) {
  const text = await file.text();
  tracksByFrame = parseTrackText(text, file.name);
  tracksName.textContent = file.name;
  inferFps();
  invalidateRender();
  render();
}

async function handleFiles(files) {
  let hasUpload = false;

  for (const file of files) {
    const lower = file.name.toLowerCase();
    if (file.type.startsWith("video/") || lower.endsWith(".mp4")) {
      hasUpload = true;
      await loadVideo(file);
    }
    if (lower.endsWith(".jsonl") || lower.endsWith(".json")) {
      hasUpload = true;
      await loadTracks(file);
    }
  }

  if (hasUpload) await playFromStart();
}

videoInput.addEventListener("change", (event) => handleFiles(event.target.files));
tracksInput.addEventListener("change", (event) => handleFiles(event.target.files));
playToggle.addEventListener("click", async () => {
  if (!video.src) return;
  if (video.paused) {
    await video.play().catch((error) => {
      console.error("Video playback failed:", error);
    });
  } else {
    video.pause();
  }
});
fpsInput.addEventListener("input", () => {
  invalidateRender();
  render();
});
opacityInput.addEventListener("input", () => {
  invalidateRender();
  render();
});
modeInput.addEventListener("change", () => {
  invalidateRender();
  render();
});
video.addEventListener("play", () => {
  playToggle.textContent = "Pause";
  startRendering();
});
video.addEventListener("pause", () => {
  playToggle.textContent = "Play";
  stopRendering();
  render();
});
video.addEventListener("loadedmetadata", () => {
  resizeCanvas();
  inferFps();
  invalidateRender();
  render();
});
video.addEventListener("seeking", () => {
  invalidateRender();
  render();
});
video.addEventListener("seeked", () => {
  invalidateRender();
  render();
  if (!video.paused) startRendering();
});
video.addEventListener("timeupdate", () => {
  if (video.currentTime < lastPlaybackTime) {
    invalidateRender();
    render();
  }
  lastPlaybackTime = video.currentTime;
});

for (const eventName of ["dragenter", "dragover"]) {
  window.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropLayer.classList.add("is-over");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  window.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropLayer.classList.remove("is-over");
  });
}

window.addEventListener("drop", (event) => handleFiles(event.dataTransfer.files));
window.addEventListener("pointermove", updatePanelFromPointer);
window.addEventListener("pointerleave", schedulePanelClose);
compactPanelMedia.addEventListener("change", () => setPanelOpen(false));
panel.addEventListener("pointerenter", () => {
  window.clearTimeout(panelHideTimer);
  setPanelOpen(true);
});
panel.addEventListener("pointerleave", schedulePanelClose);
panel.addEventListener("focusin", () => setPanelOpen(true));
panel.addEventListener("focusout", schedulePanelClose);
window.addEventListener("resize", () => {
  resizeCanvas();
  invalidateRender();
  render();
});

render();

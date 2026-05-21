const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");

const dropLayer = document.querySelector("#dropLayer");
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

function fitRect() {
  const bounds = video.getBoundingClientRect();
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

function resizeCanvas() {
  const bounds = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(bounds.width * dpr));
  canvas.height = Math.max(1, Math.floor(bounds.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function inferFps() {
  if (maxFrame && video.duration && Number.isFinite(video.duration)) {
    const fps = maxFrame / video.duration;
    fpsInput.value = fps.toFixed(2);
  }
}

function drawObject(obj, fit, opacity, mode) {
  const x = fit.x + obj.x1 * fit.scaleX;
  const y = fit.y + obj.y1 * fit.scaleY;
  const w = (obj.x2 - obj.x1) * fit.scaleX;
  const h = (obj.y2 - obj.y1) * fit.scaleY;
  const cx = fit.x + obj.center_x * fit.scaleX;
  const cy = fit.y + obj.center_y * fit.scaleY;

  ctx.fillStyle = colorForTrack(obj.track_id, opacity);
  ctx.strokeStyle = colorForTrack(obj.track_id, 0.95);
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

  ctx.fillStyle = "rgba(7, 8, 7, 0.82)";
  ctx.fillRect(x, Math.max(0, y - 24), 84, 22);
  ctx.fillStyle = "#eaff86";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillText(`${obj.class_name || "obj"} ${obj.track_id}`, x + 7, Math.max(14, y - 8));
}

function render() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const fps = Number(fpsInput.value) || 30;
  const frame = Math.max(1, Math.round(video.currentTime * fps) + 1);
  const objects = tracksByFrame.get(frame) || [];
  const fit = fitRect();
  const opacity = Number(opacityInput.value);
  const mode = modeInput.value;

  for (const obj of objects) drawObject(obj, fit, opacity, mode);

  frameStat.textContent = String(frame);
  objectStat.textContent = String(objects.length);
  trackStat.textContent = String(new Set(objects.map((obj) => obj.track_id)).size);

  rafId = requestAnimationFrame(render);
}

async function loadVideo(file) {
  video.src = URL.createObjectURL(file);
  videoName.textContent = file.name;
  dropLayer.classList.add("is-hidden");
  video.pause();
  playToggle.textContent = "Play";
}

async function loadTracks(file) {
  const text = await file.text();
  tracksByFrame = parseTrackText(text, file.name);
  tracksName.textContent = file.name;
  inferFps();
}

function handleFiles(files) {
  for (const file of files) {
    const lower = file.name.toLowerCase();
    if (file.type.startsWith("video/") || lower.endsWith(".mp4")) loadVideo(file);
    if (lower.endsWith(".jsonl") || lower.endsWith(".json")) loadTracks(file);
  }
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
video.addEventListener("play", () => {
  playToggle.textContent = "Pause";
});
video.addEventListener("pause", () => {
  playToggle.textContent = "Play";
});
video.addEventListener("loadedmetadata", () => {
  resizeCanvas();
  inferFps();
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
window.addEventListener("resize", resizeCanvas);

cancelAnimationFrame(rafId);
render();

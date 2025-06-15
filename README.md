# PTZ Simulation Control 🎥 (WIP)

> Simulates pan-tilt-zoom behavior using static camera input and OpenCV. Designed for low-cost, high-function AV setups in church/live environments.

## 🧠 Why This Exists

Professional PTZ gear is expensive. This project creates a software simulation of PTZ motion, allowing one static cam to behave dynamically — through two control modes: auto (face tracking) and manual (mouse).

---

## 🔁 Control Modes

### 🧍 Face Tracker (YOLO-based)
- Runs YOLO object detection (face-based)
- Zooms into detected face zone
- Outputs two feeds: preview (with stats), NDI (clean overlay-free)

### 🖱️ Mouse Tracker
- Manual camera simulation
- Mouse pointer acts as camera target
- Smooth zoom/pan via OpenCV
- Live preview + clean NDI feed

## 🔌 How It Works

1. Input source fed via OpenCV or NDI
2. Real-time processing depending on mode
3. Feed rendered in:
   - Preview (stats/debug window)
   - NDI output (clean full-resolution)

---

## 🚧 Challenges & Plans

- 🧠 CPU usage spikes on YOLO (no GPU)
- Plan: downscale input to 360p, upscale output to 1080p
- Merge both systems for toggle between Auto & Manual modes

---

## 🛠️ Stack

- Python, OpenCV, NDI Tools
- YOLO (face detection)
- Targeted for live streaming environments

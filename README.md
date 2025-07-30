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

## 🚀 How to Run the Face Tracking Module

### 1. 📥 Install Python

Download Python 3.8 or later from:  
https://www.python.org/downloads/

### 2. 📦 Clone the Repository

```
git clone https://github.com/Newkoncept/ptz-control.git
cd ptz-control
```

Or download the ZIP and extract it.

### 3. 🧪 Set Up a Virtual Environment

```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 4. 📚 Install Dependencies

```
pip install -r requirements.txt
```

Make sure to install the [NDI runtime](https://www.ndi.tv/tools/) for network video output support.

### 5. ▶️ Run the Application

```
python ptz_tracker.py
```

---

## 📂 Project Structure

```
ptz-control/
├── ptz_tracker.py     # Face-tracking PTZ controller (auto mode)
├── requirements.txt
└── README.md
```

---

## 🌿 Branch Overview

| Branch Name             | Purpose                             |
| ----------------------- | ----------------------------------- |
| `main`                  | Stable, production-ready builds     |
| `dev_face_tracker`      | Active development of face tracking |
| `archive_face_tracker`  | Chronological evolution (s1 → s23)  |
| `archive_mouse_tracker` | Chronological evolution (m1 - m4)   |
| `dev_mouse_tracker`     | (Planned) manual mode development   |

---

## 🗃️ Archive Note

This project began with a series of iterations (`s1.py` to `s23.py`) developed locally over months. These are preserved in the `archive_face_tracker` branch for documentation and reference. The current face tracker (`ptz_tracker.py`) is built from the final stable iteration.

---

## 🧩 Dependencies

Minimum required packages:

```
opencv-python
ultralytics
ndi-python
torch
torchvision
```

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [NewTek NDI Tools](https://www.ndi.tv/tools/)

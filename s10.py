import cv2
import time
from ultralytics import YOLO

# === Load YOLOv8 face model ===
model = YOLO("yolov8n-face.pt")

VIDEO_INDEX = 16
cap = cv2.VideoCapture(VIDEO_INDEX)
if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

cv2.namedWindow("Smart Zoom", cv2.WINDOW_NORMAL)

# === Configurable Parameters ===
ZOOM_FACTOR = 1.5
SMOOTHING_FACTOR = 0.2
FACE_LOST_TIMEOUT = 2.0
DEAD_ZONE_SCALE = 1.2

# === Internal State ===
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_visible = False
last_face_box = None
tracking_enabled = True
preset_position = None

# === Manual nudge step in pixels ===
NUDGE_STEP = 20

def adaptive_smoothing(cx, cy, smoothed_cx, smoothed_cy, dz_box):
    x1, y1, x2, y2 = dz_box
    dz_w = x2 - x1
    dz_h = y2 - y1
    near_edge = (
        abs(cx - smoothed_cx) > dz_w * 0.4 or
        abs(cy - smoothed_cy) > dz_h * 0.4
    )
    factor = 0.4 if near_edge else SMOOTHING_FACTOR
    new_cx = int(smoothed_cx * (1 - factor) + cx * factor)
    new_cy = int(smoothed_cy * (1 - factor) + cy * factor)
    return new_cx, new_cy

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed")
        break

    h, w = frame.shape[:2]

    if tracking_enabled:
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
    else:
        results = []

    face_found = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_face_box = (x1, y1, x2, y2)
            face_found = True
            break

    if tracking_enabled and face_found:
        face_visible = True
        last_seen_time = time.time()

        if smoothed_cx is None or smoothed_cy is None:
            smoothed_cx, smoothed_cy = cx, cy

        if last_face_box:
            fx1, fy1, fx2, fy2 = last_face_box
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            dz_w = int(face_w * DEAD_ZONE_SCALE)
            dz_h = int(face_h * DEAD_ZONE_SCALE)

            dz_x1 = max(smoothed_cx - dz_w // 2, 0)
            dz_y1 = max(smoothed_cy - dz_h // 2, 0)
            dz_x2 = min(smoothed_cx + dz_w // 2, w)
            dz_y2 = min(smoothed_cy + dz_h // 2, h)

            cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)

            if cx < dz_x1 or cx > dz_x2 or cy < dz_y1 or cy > dz_y2:
                smoothed_cx, smoothed_cy = adaptive_smoothing(cx, cy, smoothed_cx, smoothed_cy, (dz_x1, dz_y1, dz_x2, dz_y2))

    elif not tracking_enabled:
        face_visible = False

    show_zoomed = (face_visible or (time.time() - last_seen_time < FACE_LOST_TIMEOUT)) and smoothed_cx is not None

    if show_zoomed and smoothed_cx is not None:
        zoom_w = int(w / ZOOM_FACTOR)
        zoom_h = int(h / ZOOM_FACTOR)

        x_start = max(min(smoothed_cx - zoom_w // 2, w - zoom_w), 0)
        y_start = max(min(smoothed_cy - zoom_h // 2, h - zoom_h), 0)
        x_end = x_start + zoom_w
        y_end = y_start + zoom_h

        cropped = frame[y_start:y_end, x_start:x_end]
        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        zoomed_frame = frame.copy()

    cv2.imshow("Smart Zoom", zoomed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        ZOOM_FACTOR = min(ZOOM_FACTOR + 0.1, 3.0)
        print(f"Zoom Factor: {ZOOM_FACTOR:.1f}")
    elif key == ord('-'):
        ZOOM_FACTOR = max(ZOOM_FACTOR - 0.1, 1.1)
        print(f"Zoom Factor: {ZOOM_FACTOR:.1f}")
    elif key == ord('s'):
        tracking_enabled = False
        print("🛑 Tracking suspended.")
    elif key == ord('r'):
        tracking_enabled = True
        print("✅ Tracking resumed.")
    elif key == ord('p'):
        if smoothed_cx is not None:
            preset_position = (smoothed_cx, smoothed_cy)
            print(f"🎯 Preset saved: {preset_position}")
    elif key == ord('o'):
        if preset_position:
            smoothed_cx, smoothed_cy = preset_position
            print(f"↩️ Returned to preset: {preset_position}")
    elif key == 82:  # Arrow Up
        if smoothed_cy:
            smoothed_cy = max(smoothed_cy - NUDGE_STEP, 0)
            print("⬆️ Moved up")
    elif key == 84:  # Arrow Down
        if smoothed_cy:
            smoothed_cy = min(smoothed_cy + NUDGE_STEP, h)
            print("⬇️ Moved down")
    elif key == 81:  # Arrow Left
        if smoothed_cx:
            smoothed_cx = max(smoothed_cx - NUDGE_STEP, 0)
            print("⬅️ Moved left")
    elif key == 83:  # Arrow Right
        if smoothed_cx:
            smoothed_cx = min(smoothed_cx + NUDGE_STEP, w)
            print("➡️ Moved right")

cap.release()
cv2.destroyAllWindows()

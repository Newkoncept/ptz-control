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
DEAD_ZONE_SCALE = 6  # Dead zone around the center is 20% wider than face box
FACE_LOST_TIMEOUT = 1.0  # Seconds to hold last frame after face disappears

# === Internal State ===
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_visible = False
last_face_box = None  # (x1, y1, x2, y2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed")
        break

    h, w = frame.shape[:2]
    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    face_found = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_face_box = (x1, y1, x2, y2)
            face_found = True
            break

    if face_found:
        face_visible = True
        last_seen_time = time.time()

        # Initialize tracking center
        if smoothed_cx is None or smoothed_cy is None:
            smoothed_cx, smoothed_cy = cx, cy

        # Create dead zone box around current smoothed center
        if last_face_box:
            face_w = last_face_box[2] - last_face_box[0]
            face_h = last_face_box[3] - last_face_box[1]
            dz_w = int(face_w * DEAD_ZONE_SCALE)
            dz_h = int(face_h * DEAD_ZONE_SCALE)

            dz_x1 = max(smoothed_cx - dz_w // 2, 0)
            dz_y1 = max(smoothed_cy - dz_h // 2, 0)
            dz_x2 = min(smoothed_cx + dz_w // 2, w)
            dz_y2 = min(smoothed_cy + dz_h // 2, h)

            # Draw dead zone box
            cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)

            # Only update if face is outside dead zone
            if cx < dz_x1 or cx > dz_x2:
                smoothed_cx = int(smoothed_cx * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
            if cy < dz_y1 or cy > dz_y2:
                smoothed_cy = int(smoothed_cy * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)

    else:
        face_visible = False

    show_zoomed = face_visible or (time.time() - last_seen_time < FACE_LOST_TIMEOUT and smoothed_cx is not None)

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
        smoothed_cx, smoothed_cy = None, None
        zoomed_frame = frame.copy()

    cv2.imshow("Smart Zoom", zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

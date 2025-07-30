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
DEAD_ZONE_SCALE = 1.2
FACE_LOST_TIMEOUT = 2.0
PAN_STEP = 20  # Manual pan step (pixels)

# === Internal State ===
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_visible = False
last_face_box = None
tracking_enabled = True
manual_offset_x, manual_offset_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed")
        break

    h, w = frame.shape[:2]
    current_time = time.time()

    face_found = False
    cx, cy = None, None
    if tracking_enabled:
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
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
        last_seen_time = current_time

        if smoothed_cx is None or smoothed_cy is None:
            smoothed_cx, smoothed_cy = cx, cy

        # Create dead zone box around smoothed center
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

            # Update only if face outside dead zone
            if cx < dz_x1 or cx > dz_x2:
                smoothed_cx = int(smoothed_cx * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
            if cy < dz_y1 or cy > dz_y2:
                smoothed_cy = int(smoothed_cy * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)

    elif tracking_enabled:
        face_visible = False

    # === Zoom logic ===
    show_zoomed = tracking_enabled and (
        (face_visible or (current_time - last_seen_time < FACE_LOST_TIMEOUT))
        and smoothed_cx is not None
    )

    if show_zoomed:
        zoom_w = int(w / ZOOM_FACTOR)
        zoom_h = int(h / ZOOM_FACTOR)

        center_x = smoothed_cx + manual_offset_x
        center_y = smoothed_cy + manual_offset_y

        x_start = max(min(center_x - zoom_w // 2, w - zoom_w), 0)
        y_start = max(min(center_y - zoom_h // 2, h - zoom_h), 0)
        x_end = x_start + zoom_w
        y_end = y_start + zoom_h

        cropped = frame[y_start:y_end, x_start:x_end]
        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        smoothed_cx, smoothed_cy = None, None
        manual_offset_x, manual_offset_y = 0, 0
        zoomed_frame = frame.copy()

    cv2.imshow("Smart Zoom", zoomed_frame)

    # key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        tracking_enabled = False
        smoothed_cx, smoothed_cy = None, None
        print("🛑 Tracking suspended.")
    elif key == ord('r'):
        tracking_enabled = True
        print("▶️ Tracking resumed.")
    elif key == ord('1'):  # Top-left
        smoothed_cx, smoothed_cy = w // 4, h // 4
        tracking_enabled = False
    elif key == ord('2'):  # Top-right
        smoothed_cx, smoothed_cy = 3 * w // 4, h // 4
        tracking_enabled = False
    elif key == ord('3'):  # Bottom-left
        smoothed_cx, smoothed_cy = w // 4, 3 * h // 4
        tracking_enabled = False
    elif key == ord('4'):  # Bottom-right
        smoothed_cx, smoothed_cy = 3 * w // 4, 3 * h // 4
        tracking_enabled = False
    # elif key == 82:  # Up arrow
    elif key == 2490368:  # Up arrow
        manual_offset_y -= PAN_STEP
        print("⬆️ Moved up")
    # elif key == 84:  # Down arrow
    elif key == 2621440:  # Down arrow
        manual_offset_y += PAN_STEP
        print("⬇️ Moved down")
    # elif key == 81:  # Left arrow
    elif key == 2424832:  # Left arrow
        manual_offset_x -= PAN_STEP
        print("⬅️ Moved left")
    # elif key == 83:  # Right arrow
    elif key == 2555904:  # Right arrow
        manual_offset_x += PAN_STEP
        print("➡️ Moved right")


cap.release()
cv2.destroyAllWindows()

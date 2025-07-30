import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 face model
model = YOLO("yolov8n-face.pt")

VIDEO_INDEX = 16
cap = cv2.VideoCapture(VIDEO_INDEX)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

WINDOW_NAME = "Smart Zoom with Panning"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# === Configurable Parameters ===
ZOOM_FACTOR = 1.5
SMOOTHING_FACTOR = 0.2
MARGIN_X = 0.15
MARGIN_Y = 0.2
FACE_LOST_TIMEOUT = 1.5  # Hold last frame for 1 second after face disappears
DEAD_ZONE_RADIUS = 15  # Pixels; tweak based on resolution and smoothness

# === Internal State ===
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_currently_visible = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, exiting...")
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
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            face_found = True
            break  # Only one face for now

    if face_found:
        face_currently_visible = True
        last_seen_time = time.time()

        if smoothed_cx is None:
            smoothed_cx, smoothed_cy = cx, cy
        else:

            # Calculate movement delta
            dx = cx - smoothed_cx
            dy = cy - smoothed_cy
            distance_squared = dx * dx + dy * dy

            if distance_squared > DEAD_ZONE_RADIUS ** 2:
                # Only update if outside the dead zone
                smoothed_cx += int(dx * SMOOTHING_FACTOR)
                smoothed_cy += int(dy * SMOOTHING_FACTOR)


            # margin_left = int(w * (0.5 - MARGIN_X))
            # margin_right = int(w * (0.5 + MARGIN_X))
            # margin_top = int(h * (0.5 - MARGIN_Y))
            # margin_bottom = int(h * (0.5 + MARGIN_Y))

            # if cx < margin_left or cx > margin_right:
            #     smoothed_cx = int(smoothed_cx * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
            # if cy < margin_top or cy > margin_bottom:
            #     smoothed_cy = int(smoothed_cy * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)

            # Always update smoothed position (continuous smooth panning)
            smoothed_cx = int(smoothed_cx * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
            smoothed_cy = int(smoothed_cy * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)


    else:
        face_currently_visible = False

    # Decide what to show
    if face_currently_visible or (time.time() - last_seen_time < FACE_LOST_TIMEOUT and smoothed_cx is not None):
        zoom_w = int(w / ZOOM_FACTOR)
        zoom_h = int(h / ZOOM_FACTOR)

        x_start = max(min(smoothed_cx - zoom_w // 2, w - zoom_w), 0)
        y_start = max(min(smoothed_cy - zoom_h // 2, h - zoom_h), 0)
        x_end = x_start + zoom_w
        y_end = y_start + zoom_h

        cropped = frame[y_start:y_end, x_start:x_end]
        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Reset smoothed values
        smoothed_cx, smoothed_cy = None, None
        zoomed_frame = frame.copy()

    # if smoothed_cx is not None and smoothed_cy is not None:
    #     dz_color = (255, 0, 255)  # Purple box
    #     dz_thickness = 2
    #     x_dz1 = smoothed_cx - DEAD_ZONE_RADIUS
    #     y_dz1 = smoothed_cy - DEAD_ZONE_RADIUS
    #     x_dz2 = smoothed_cx + DEAD_ZONE_RADIUS
    #     y_dz2 = smoothed_cy + DEAD_ZONE_RADIUS
    #     cv2.rectangle(frame, (x_dz1, y_dz1), (x_dz2, y_dz2), dz_color, dz_thickness)

    cv2.imshow(WINDOW_NAME, zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

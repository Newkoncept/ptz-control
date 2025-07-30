import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")

VIDEO_INDEX = 16
cap = cv2.VideoCapture(VIDEO_INDEX)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

WINDOW_NAME = "Smoothed Zoom on Face"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

ZOOM_FACTOR = 1.5

# Initialize smoothed center coordinates
smoothed_cx = None
smoothed_cy = None
SMOOTHING_FACTOR = 0.2  # Lower = smoother (0.05–0.2 is typical)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, exiting...")
        break

    h, w = frame.shape[:2]
    zoomed_frame = frame.copy()

    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    face_found = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            face_found = True
            break

    if face_found:
        # Initialize smoothed position
        if smoothed_cx is None:
            smoothed_cx, smoothed_cy = cx, cy
        else:
            # Apply exponential moving average
            smoothed_cx = int(smoothed_cx * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR)
            smoothed_cy = int(smoothed_cy * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR)

        # Define crop area
        zoom_w = int(w / ZOOM_FACTOR)
        zoom_h = int(h / ZOOM_FACTOR)

        x_start = max(smoothed_cx - zoom_w // 2, 0)
        y_start = max(smoothed_cy - zoom_h // 2, 0)
        x_end = min(x_start + zoom_w, w)
        y_end = min(y_start + zoom_h, h)

        # Crop and resize
        cropped = frame[y_start:y_end, x_start:x_end]
        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # No face found: keep previous zoom or show full frame
        smoothed_cx = None
        smoothed_cy = None
        zoomed_frame = cv2.resize(frame, (w, h))

    cv2.imshow(WINDOW_NAME, zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

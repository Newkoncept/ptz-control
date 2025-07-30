import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")

VIDEO_INDEX = 16  # Adjust to your camera index
cap = cv2.VideoCapture(VIDEO_INDEX)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

WINDOW_NAME = "Gentle Zoom on Face"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

ZOOM_FACTOR = 1.5  # Change between 1.2 - 2.0 for stronger zoom

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, exiting...")
        break

    h, w = frame.shape[:2]
    zoomed_frame = frame.copy()

    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Get face center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determine size of crop based on zoom factor
            zoom_w = int(w / ZOOM_FACTOR)
            zoom_h = int(h / ZOOM_FACTOR)

            # Ensure crop box stays within frame
            x_start = max(cx - zoom_w // 2, 0)
            y_start = max(cy - zoom_h // 2, 0)
            x_end = min(x_start + zoom_w, w)
            y_end = min(y_start + zoom_h, h)

            # Crop and resize
            cropped = frame[y_start:y_end, x_start:x_end]
            zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            break  # Use only the first detected face

    cv2.imshow(WINDOW_NAME, zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

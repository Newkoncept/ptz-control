import cv2
from ultralytics import YOLO

# Load the YOLOv8 face detection model (replace with your .pt file path)
# model = YOLO("yolo/yolov8n.pt")  # Ensure this model detects faces, not general objects
model = YOLO("yolov8n-face.pt")  # Ensure this model detects faces, not general objects

# Open video stream (0 = default webcam, 1+ = other cameras or OBS virtual cam)
VIDEO_INDEX = 16  # Change to match your input source (0, 1, etc.)
cap = cv2.VideoCapture(VIDEO_INDEX)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

# Set a larger display window
WINDOW_NAME = "YOLOv8 Face Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(WINDOW_NAME, 1280, 720)
# cv2.resizeWindow(WINDOW_NAME, 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, exiting...")
        break

    # Run face detection with YOLO
    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

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

    cv2.imshow(WINDOW_NAME, frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

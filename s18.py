import time

start = time.time()
print("⏳ Importing modules...")
import cv2
import threading
import queue
from ultralytics import YOLO
import NDIlib as ndi

print("✅ Modules loaded in", round(time.time() - start, 2), "s")

# === Configuration ===
CAMERA_INDEX = 7                 # Your webcam index
DETECTION_RES = (640, 360)       # Detection happens at this size
OUTPUT_RES = (1920, 1080)        # Final video output resolution
INFERENCE_CONF = 0.4             # Detection confidence threshold
ZOOM_FACTOR = 1.5                # How much we zoom into the detected face
SMOOTHING_FACTOR = 0.2
HUD_ENABLED = True
TRACKER_BOX = True

# === Load model ===
model = YOLO("yolov8n-face.pt")
print("✅ YOLO model loaded in", round(time.time() - start, 2), "s")


# === Shared queue between capture and processing ===
frame_queue = queue.Queue(maxsize=3)

# === Initialize NDI ===
if not ndi.initialize():
    raise Exception("NDI initialization failed")
send_settings = ndi.SendCreate()
send_settings.ndi_name = "Virtual PTZ"
ndi_sender = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

# === Frame capture thread ===
def capture_thread():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_RES[1])

    while True:
        ret, full_frame = cap.read()
        if not ret:
            continue

        detection_frame = cv2.resize(full_frame, DETECTION_RES)
        frame_queue.put((full_frame, detection_frame))

capture_worker = threading.Thread(target=capture_thread, daemon=True)
capture_worker.start()

# === Initialize state ===
smoothed_cx, smoothed_cy = None, None
last_seen = time.time()
face_box = None

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

# === Processing loop ===
while True:
    if frame_queue.empty():
        time.sleep(0.01)
        continue

    full_frame, det_frame = frame_queue.get()
    hud_frame = det_frame.copy()

    # Run detection
    results = model.predict(det_frame, imgsz=640, conf=INFERENCE_CONF, verbose=False)

    h_scale = full_frame.shape[1] / DETECTION_RES[0]
    v_scale = full_frame.shape[0] / DETECTION_RES[1]

    found = False
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                face_box = (
                    int(x1 * h_scale), int(y1 * v_scale),
                    int(x2 * h_scale), int(y2 * v_scale)
                )
                smoothed_cx = cx if smoothed_cx is None else int((1 - SMOOTHING_FACTOR) * smoothed_cx + SMOOTHING_FACTOR * cx)
                smoothed_cy = cy if smoothed_cy is None else int((1 - SMOOTHING_FACTOR) * smoothed_cy + SMOOTHING_FACTOR * cy)
                found = True
                break
        if found:
            break

    if found:
        last_seen = time.time()

    # Calculate zoomed region from 1080p frame
    if smoothed_cx is not None and smoothed_cy is not None:
        zoom_w = int(full_frame.shape[1] / ZOOM_FACTOR)
        zoom_h = int(full_frame.shape[0] / ZOOM_FACTOR)
        cx = int(smoothed_cx * h_scale)
        cy = int(smoothed_cy * v_scale)
        x1 = max(cx - zoom_w // 2, 0)
        y1 = max(cy - zoom_h // 2, 0)
        x2 = min(x1 + zoom_w, full_frame.shape[1])
        y2 = min(y1 + zoom_h, full_frame.shape[0])
        cropped = full_frame[y1:y2, x1:x2]
        output_frame = cv2.resize(cropped, OUTPUT_RES)
    else:
        output_frame = cv2.resize(full_frame, OUTPUT_RES)

    # Send via NDI
    bgra = cv2.cvtColor(output_frame, cv2.COLOR_BGR2BGRA)
    video_frame.data = bgra
    video_frame.xres = OUTPUT_RES[0]
    video_frame.yres = OUTPUT_RES[1]
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    video_frame.line_stride_in_bytes = OUTPUT_RES[0] * 4
    ndi.send_send_video_v2(ndi_sender, video_frame)

    # === HUD Preview ===
    if HUD_ENABLED:
        if face_box and TRACKER_BOX:
            x1, y1, x2, y2 = face_box
            x1 = int(x1 / h_scale)
            y1 = int(y1 / v_scale)
            x2 = int(x2 / h_scale)
            y2 = int(y2 / v_scale)
            cv2.rectangle(hud_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(hud_frame, f"Tracking: {'Yes' if found else 'No'}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found else (0, 0, 255), 2)

    cv2.imshow("Preview", hud_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ndi.send_destroy(ndi_sender)
ndi.destroy()
cv2.destroyAllWindows()

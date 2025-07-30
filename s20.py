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
CAMERA_INDEX = 7                    # Your webcam index
# DETECTION_RES = (640, 360)          # Detection happens at this size
DETECTION_RES = (320, 180)          # Detection happens at this size
OUTPUT_RES = (1920, 1080)           # Final video output resolution
INFERENCE_CONF = 0.4                # Detection confidence threshold
MAX_ZOOM = 3.0
MIN_ZOOM = 1.5
MAX_DEAD_ZONE = 10.0
MIN_DEAD_ZONE = 2.0
MAX_SMOOTHING_FACTOR = 0.5
MIN_SMOOTHING_FACTOR = 0.05


operation_parameters = {
    "zoom_factor": 1.5,             # How much we zoom into the detected face
    "smoothing_factor": 0.2,
    "dead_zone_scale": 5.0,
    "hud_enabled": True,
    "tracking_enabled": True
}



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
print("✅ ndi initialized in", round(time.time() - start, 2), "s")


def HUD_Display(frame, message, x_position, y_position, color):
    cv2.putText(frame, message, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def handle_key_press(key):
    if key == ord('q'):
        return 'exit'
    elif key == ord('+'):
        operation_parameters["zoom_factor"] = min(operation_parameters["zoom_factor"] + 0.1, MAX_ZOOM)
    elif key == ord('-'):
        operation_parameters["zoom_factor"] = max(operation_parameters["zoom_factor"] - 0.1, MIN_ZOOM)
    elif key == ord('n'):
        operation_parameters["dead_zone_scale"] = min(operation_parameters["dead_zone_scale"] + 1.0, MAX_DEAD_ZONE)
    elif key == ord('m'):
        operation_parameters["dead_zone_scale"] = max(operation_parameters["dead_zone_scale"] - 1.0, MIN_DEAD_ZONE)
    elif key == ord('j'):
        operation_parameters["smoothing_factor"] = min(operation_parameters["smoothing_factor"] + 0.01, MAX_SMOOTHING_FACTOR)
    elif key == ord('k'):
        operation_parameters["smoothing_factor"] = max(operation_parameters["smoothing_factor"] - 0.01, MIN_SMOOTHING_FACTOR)
    elif key == ord('h'):
        operation_parameters["hud_enabled"] = not operation_parameters["hud_enabled"]
    elif key == ord('f'):
        operation_parameters["tracking_enabled"] = not operation_parameters["tracking_enabled"]

    return 'continue'

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

print("✅ capture worker initialized in", round(time.time() - start, 2), "s")


# === Initialize state ===
smoothed_cx, smoothed_cy = None, None
last_seen = time.time()
face_box = None

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)


print("✅ Program start initialized in", round(time.time() - start, 2), "s")
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

                smoothed_cx = cx if smoothed_cx is None else smoothed_cx
                smoothed_cy = cy if smoothed_cy is None else smoothed_cy

                # Calculate dead zone relative to smoothed position
                face_width = x2 - x1
                face_height = y2 - y1
                dz_w = int(face_width * operation_parameters["dead_zone_scale"])
                dz_h = int(face_height * operation_parameters["dead_zone_scale"])

                dz_x1 = smoothed_cx - dz_w // 2
                dz_y1 = smoothed_cy - dz_h // 2
                dz_x2 = smoothed_cx + dz_w // 2
                dz_y2 = smoothed_cy + dz_h // 2

                if cx < dz_x1 or cx > dz_x2:
                    smoothed_cx = int((1 - operation_parameters["smoothing_factor"]) * smoothed_cx + operation_parameters["smoothing_factor"] * cx)
                if cy < dz_y1 or cy > dz_y2:
                    smoothed_cy = int((1 - operation_parameters["smoothing_factor"]) * smoothed_cy + operation_parameters["smoothing_factor"] * cy)

                found = True
                break
        if found:
            break

    if found:
        last_seen = time.time()

    # Calculate zoomed region from 1080p frame
    if smoothed_cx is not None and smoothed_cy is not None:
        zoom_w = int(full_frame.shape[1] / operation_parameters["zoom_factor"])
        zoom_h = int(full_frame.shape[0] / operation_parameters["zoom_factor"])
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

    # === Tracking Box ===
    if face_box and operation_parameters["tracking_enabled"]:
        x1, y1, x2, y2 = face_box
        x1 = int(x1 / h_scale)
        y1 = int(y1 / v_scale)
        x2 = int(x2 / h_scale)
        y2 = int(y2 / v_scale)
        cv2.rectangle(hud_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw dead zone relative to smoothed position
        dz_x1_disp = int(dz_x1)
        dz_y1_disp = int(dz_y1)
        dz_x2_disp = int(dz_x2)
        dz_y2_disp = int(dz_y2)
        cv2.rectangle(hud_frame, (dz_x1_disp, dz_y1_disp), (dz_x2_disp, dz_y2_disp), (255, 0, 255), 2)

    # === HUD Preview ===
    if operation_parameters["hud_enabled"]:
        hud_color = (0, 255, 0) if operation_parameters["tracking_enabled"] else (0, 0, 255)
        y_offset = 50

        HUD_Display(hud_frame, f"Tracking: {'ON' if found else 'OFF'}", 10, y_offset, hud_color)
        for key, val in operation_parameters.items():
            if key == "tracking_enabled":
                continue
            y_offset += 25
            HUD_Display(hud_frame, f"{key}: {val}", 10, y_offset, hud_color)


    cv2.imshow("Preview", hud_frame)
    
    # === Key Controls ===
    key = cv2.waitKey(1) & 0xFF
    # key = cv2.waitKey(20) & 0xFF
    if handle_key_press(key) == 'exit':
        break
    

ndi.send_destroy(ndi_sender)
ndi.destroy()
cv2.destroyAllWindows()

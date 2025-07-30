import time
import cv2
import threading
import queue
from ultralytics import YOLO
import NDIlib as ndi

start = time.time()
print("⏳ Importing modules...")

# === Configuration ===
CAMERA_INDEX = 0
DETECTION_RES = (320, 180)
OUTPUT_RES = (1920, 1080)
INFERENCE_CONF = 0.4
MAX_ZOOM = 3.0
MIN_ZOOM = 1.0
MAX_DEAD_ZONE = 10.0
MIN_DEAD_ZONE = 2.0
MAX_SMOOTHING_FACTOR = 0.5
MIN_SMOOTHING_FACTOR = 0.05
ZOOM_SMOOTHING = 0.05
DETECTION_INTERVAL = 5

operation_parameters = {
    "zoom_factor": 1.5,
    "smoothing_factor": 0.2,
    "dead_zone_scale": 5.0,
    "hud_enabled": True,
    "tracking_enabled": True
}

model = YOLO("yolov8n-face.pt")
print("✅ YOLO model loaded in", round(time.time() - start, 2), "s")

frame_queue = queue.Queue(maxsize=3)

if not ndi.initialize():
    raise Exception("NDI initialization failed")
send_settings = ndi.SendCreate()
send_settings.ndi_name = "Virtual PTZ"
di_sender = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()
print("✅ ndi initialized in", round(time.time() - start, 2), "s")

def HUD_Display(frame, message, x, y, color):
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

threading.Thread(target=capture_thread, daemon=True).start()
print("✅ capture worker initialized in", round(time.time() - start, 2), "s")

smoothed_cx, smoothed_cy = None, None
face_box = None
current_zoom = 1.0
frame_counter = 0
cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
print("✅ Program start initialized in", round(time.time() - start, 2), "s")

while True:
    if frame_queue.empty():
        time.sleep(0.01)
        continue

    full_frame, det_frame = frame_queue.get()
    hud_frame = det_frame.copy()
    frame_counter += 1

    h_scale = full_frame.shape[1] / DETECTION_RES[0]
    v_scale = full_frame.shape[0] / DETECTION_RES[1]

    if operation_parameters["tracking_enabled"]:
        if frame_counter % DETECTION_INTERVAL == 0 or face_box is None:
            try:
                results = model.predict(det_frame, imgsz=DETECTION_RES[0], conf=INFERENCE_CONF, verbose=False)
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            cx_full = int(cx * h_scale)
                            cy_full = int(cy * v_scale)

                            face_box = (
                                int(x1 * h_scale), int(y1 * v_scale),
                                int(x2 * h_scale), int(y2 * v_scale)
                            )

                            if smoothed_cx is None:
                                smoothed_cx, smoothed_cy = cx_full, cy_full
                            break
                    break
            except:
                face_box = None

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
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

    target_zoom = operation_parameters["zoom_factor"] if operation_parameters["tracking_enabled"] else 1.0
    current_zoom = (1 - ZOOM_SMOOTHING) * current_zoom + ZOOM_SMOOTHING * target_zoom

    zoom_w = int(full_frame.shape[1] / current_zoom)
    zoom_h = int(full_frame.shape[0] / current_zoom)

    cx = smoothed_cx if smoothed_cx is not None else full_frame.shape[1] // 2
    cy = smoothed_cy if smoothed_cy is not None else full_frame.shape[0] // 2

    x1 = max(0, min(cx - zoom_w // 2, full_frame.shape[1] - zoom_w))
    y1 = max(0, min(cy - zoom_h // 2, full_frame.shape[0] - zoom_h))
    x2 = x1 + zoom_w
    y2 = y1 + zoom_h
    cropped = full_frame[y1:y2, x1:x2]
    output_frame = cv2.resize(cropped, OUTPUT_RES)

    bgra = cv2.cvtColor(output_frame, cv2.COLOR_BGR2BGRA)
    video_frame.data = bgra
    video_frame.xres = OUTPUT_RES[0]
    video_frame.yres = OUTPUT_RES[1]
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    video_frame.line_stride_in_bytes = OUTPUT_RES[0] * 4
    ndi.send_send_video_v2(di_sender, video_frame)

    if face_box and operation_parameters["tracking_enabled"]:
        x1, y1, x2, y2 = face_box
        scale_down_x = DETECTION_RES[0] / OUTPUT_RES[0]
        scale_down_y = DETECTION_RES[1] / OUTPUT_RES[1]
        cv2.rectangle(hud_frame, (int(x1 * scale_down_x), int(y1 * scale_down_y)), (int(x2 * scale_down_x), int(y2 * scale_down_y)), (0, 255, 0), 2)
        cv2.rectangle(hud_frame, (int(dz_x1 * scale_down_x), int(dz_y1 * scale_down_y)), (int(dz_x2 * scale_down_x), int(dz_y2 * scale_down_y)), (255, 0, 255), 2)

    if operation_parameters["hud_enabled"]:
        hud_color = (0, 255, 0) if operation_parameters["tracking_enabled"] else (0, 0, 255)
        y_offset = 50
        HUD_Display(hud_frame, f"Tracking: {'ON' if operation_parameters['tracking_enabled'] else 'OFF'}", 10, y_offset, hud_color)
        for key, val in operation_parameters.items():
            y_offset += 25
            HUD_Display(hud_frame, f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}", 10, y_offset, hud_color)

    cv2.imshow("Preview", hud_frame)
    key = cv2.waitKey(1) & 0xFF
    if handle_key_press(key) == 'exit':
        break

ndi.send_destroy(di_sender)
ndi.destroy()
cv2.destroyAllWindows()

import cv2
import time
import NDIlib as ndi
from ultralytics import YOLO

# === Load YOLOv8 Face Detection Model ===
model = YOLO("yolov8n-face.pt")

# === Video Capture Setup ===
VIDEO_INDEX = 6
cap = cv2.VideoCapture(VIDEO_INDEX)
if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

# Set capture resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.setUseOptimized(True)

WINDOW_NAME = "Smart Zoom"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# === Configurable Parameters ===
conf_parameters = {
    "zoom_factor": 1.5,
    "smoothing_factor": 0.2,
    "dead_zone_scale": 1.2,
    "face_lost_timeout": 2.0,
    "show_tracker": True,
    "show_hud": True,
    "inference_interval": 3  # Run inference every N frames
}

# === Internal State ===
frame_count = 0
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_visible = False
last_face_box = None
tracking_enabled = True
manual_offset_x, manual_offset_y = 0, 0
transition_frames = 15
transition_counter = 0
prev_crop_box = None
current_crop_box = None

# === NDI Initialization ===
if not ndi.initialize():
    raise Exception("NDI initialization failed")

send_settings = ndi.SendCreate()
send_settings.ndi_name = "SmartZoom NDI"
ndi_sender = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

def HUD_Display(frame, message, x_position, y_position, color):
    cv2.putText(frame, message, (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def lerp(a, b, t):
    return int(a + (b - a) * t)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed")
        break

    hud_frame = frame.copy()


    frame_count += 1
    canvas_height, canvas_width = frame.shape[:2]
    current_time = time.time()
    face_found = False
    cx, cy = None, None

    # === Inference Control ===
    run_detection = tracking_enabled and (frame_count % conf_parameters["inference_interval"] == 0)

    if run_detection:
        # Downscale for faster inference
        small_frame = cv2.resize(frame, (640, 360))
        results = model.predict(small_frame, imgsz=640, conf=0.4, verbose=False)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)

                # Rescale to original resolution
                x1 = int(x1 * 3)
                y1 = int(y1 * 3)
                x2 = int(x2 * 3)
                y2 = int(y2 * 3)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                last_face_box = (x1, y1, x2, y2, conf)
                face_found = True
                break

    if face_found:
        face_visible = True
        last_seen_time = current_time


        if smoothed_cx is None or smoothed_cy is None:
            smoothed_cx, smoothed_cy = cx, cy

        face_width = last_face_box[2] - last_face_box[0]
        face_height = last_face_box[3] - last_face_box[1]
        deadzone_w = int(face_width * conf_parameters["dead_zone_scale"])
        deadzone_h = int(face_height * conf_parameters["dead_zone_scale"])

        dz_x1 = max(smoothed_cx - deadzone_w // 2, 0)
        dz_y1 = max(smoothed_cy - deadzone_h // 2, 0)
        dz_x2 = min(smoothed_cx + deadzone_w // 2, canvas_width)
        dz_y2 = min(smoothed_cy + deadzone_h // 2, canvas_height)

        if conf_parameters["show_tracker"]:
            cv2.rectangle(hud_frame, (last_face_box[0], last_face_box[1]),
                          (last_face_box[2], last_face_box[3]), (0, 255, 0), 2)
            cv2.rectangle(hud_frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 0, 255), 2)

        if cx < dz_x1 or cx > dz_x2:
            smoothed_cx = int(smoothed_cx * (1 - conf_parameters["smoothing_factor"]) + cx * conf_parameters["smoothing_factor"])
        if cy < dz_y1 or cy > dz_y2:
            smoothed_cy = int(smoothed_cy * (1 - conf_parameters["smoothing_factor"]) + cy * conf_parameters["smoothing_factor"])

    elif tracking_enabled:
        face_visible = False

    # === Zoom Region Computation ===
    show_zoomed = tracking_enabled and (
        (face_visible or (current_time - last_seen_time < conf_parameters["face_lost_timeout"])) 
        and smoothed_cx is not None
    )

    if show_zoomed:
        zoom_w = int(canvas_width / conf_parameters["zoom_factor"])
        zoom_h = int(canvas_height / conf_parameters["zoom_factor"])
        cx, cy = smoothed_cx + manual_offset_x, smoothed_cy + manual_offset_y

        x1 = max(min(cx - zoom_w // 2, canvas_width - zoom_w), 0)
        y1 = max(min(cy - zoom_h // 2, canvas_height - zoom_h), 0)
        x2 = x1 + zoom_w
        y2 = y1 + zoom_h
        target_crop_box = (x1, y1, x2, y2)
    else:
        target_crop_box = (0, 0, canvas_width, canvas_height)
        smoothed_cx, smoothed_cy = None, None
        manual_offset_x, manual_offset_y = 0, 0

    if current_crop_box != target_crop_box:
        prev_crop_box = current_crop_box if current_crop_box else target_crop_box
        current_crop_box = target_crop_box
        transition_counter = transition_frames

    if transition_counter > 0 and prev_crop_box:
        t = 1 - (transition_counter / transition_frames)
        x1 = lerp(prev_crop_box[0], current_crop_box[0], t)
        y1 = lerp(prev_crop_box[1], current_crop_box[1], t)
        x2 = lerp(prev_crop_box[2], current_crop_box[2], t)
        y2 = lerp(prev_crop_box[3], current_crop_box[3], t)
        transition_counter -= 1
    else:
        x1, y1, x2, y2 = current_crop_box



    # === Frame Zoom and NDI Send ===
    cropped = frame[y1:y2, x1:x2]
    hud_cropped = hud_frame[y1:y2, x1:x2] # HUD version for preview

    zoomed = cv2.resize(cropped, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
    preview_zoomed = cv2.resize(hud_cropped, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)


    # final_frame = cv2.resize(zoomed, (1920, 1080))
    # bgra = cv2.cvtColor(final_frame, cv2.COLOR_BGR2BGRA)
    bgra = cv2.cvtColor(zoomed, cv2.COLOR_BGR2BGRA)

    video_frame.xres = 1920
    video_frame.yres = 1080
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    video_frame.line_stride_in_bytes = 1920 * 4
    video_frame.data = bgra
    ndi.send_send_video_v2(ndi_sender, video_frame)

    # === HUD Overlay ===
    hud_color = (0, 255, 0) if tracking_enabled else (0, 0, 255)
    y_pos = 30

    if conf_parameters["show_hud"]:
        HUD_Display(preview_zoomed, f"Tracking: {'ON' if tracking_enabled else 'OFF'}", 10, y_pos, hud_color)
        for key, val in conf_parameters.items():
            y_pos += 25
            HUD_Display(preview_zoomed, f"{key}: {val}", 10, y_pos, hud_color)

        if not face_visible and (current_time - last_seen_time > conf_parameters["face_lost_timeout"]):
            HUD_Display(preview_zoomed, "🔍 Face Lost", 10, y_pos + 25, (0, 0, 255))

    cv2.imshow(WINDOW_NAME, preview_zoomed)

    # === Key Controls ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
    elif key == ord('f'):
        conf_parameters["show_tracker"] = not conf_parameters["show_tracker"]
    elif key == ord('h'):
        conf_parameters["show_hud"] = not conf_parameters["show_hud"]
    elif key == ord('+'):
        conf_parameters["zoom_factor"] = min(conf_parameters["zoom_factor"] + 0.1, 3.0)
    elif key == ord('-'):
        conf_parameters["zoom_factor"] = max(conf_parameters["zoom_factor"] - 0.1, 1.1)
    elif key == ord('n'):
        conf_parameters["dead_zone_scale"] = min(conf_parameters["dead_zone_scale"] + 1.0, 10.0)
    elif key == ord('m'):
        conf_parameters["dead_zone_scale"] = max(conf_parameters["dead_zone_scale"] - 1.0, 1.2)
    elif key == ord('j'):
        conf_parameters["smoothing_factor"] = min(conf_parameters["smoothing_factor"] + 0.01, 0.2)
    elif key == ord('k'):
        conf_parameters["smoothing_factor"] = max(conf_parameters["smoothing_factor"] - 0.01, 0.05)

cap.release()
cv2.destroyAllWindows()
ndi.send_destroy(ndi_sender)
ndi.destroy()

import cv2
import time
import NDIlib as ndi  # New
from ultralytics import YOLO

# === Load YOLOv8 face model ===
model = YOLO("yolov8n-face.pt")

VIDEO_INDEX = 6
cap = cv2.VideoCapture(VIDEO_INDEX)
if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

WINDOW_NAME = "Smart Zoom"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# === Configurable Parameters ===
conf_parameters = {
    "zoom_factor" : 1.5,
    "smoothing_factor" : 0.2,
    "dead_zone_scale" : 1.2,
    "face_lost_timeout" : 2.0,
    "show_tracker" : True,
    "show_hud": True
}

# === Internal State ===
smoothed_cx, smoothed_cy = None, None
last_seen_time = time.time()
face_visible = False
last_face_box = None
tracking_enabled = True
manual_offset_x, manual_offset_y = 0, 0

    # === Zoom transition state ===
transition_frames = 15
transition_counter = 0
prev_crop_box = None
current_crop_box = None



def HUD_Display(frame, message, x_position, y_position, color):
    cv2.putText(frame, message, (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
def lerp(a, b, t):
    return int(a + (b - a) * t)


# === NDI Initialization ===
if not ndi.initialize():
    raise Exception("NDI initialization failed")

send_settings = ndi.SendCreate()
send_settings.ndi_name = "SmartZoom NDI"
ndi_sender = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed")
        break

    canvas_height, canvas_width = frame.shape[:2]

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

                last_face_box = (x1, y1, x2, y2, conf)
                face_found = True
                break

    if face_found:
        face_visible = True
        last_seen_time = current_time


        if smoothed_cx is None or smoothed_cy is None:
            smoothed_cx, smoothed_cy = cx, cy

        # Create dead zone box around smoothed center
        if last_face_box:
            face_width = last_face_box[2] - last_face_box[0]
            face_height = last_face_box[3] - last_face_box[1]
            deadzone_width = int(face_width * conf_parameters["dead_zone_scale"])
            deadzone_height = int(face_height * conf_parameters["dead_zone_scale"])

            deadzone_x1 = max(smoothed_cx - deadzone_width // 2, 0)
            deadzone_y1 = max(smoothed_cy - deadzone_height // 2, 0)
            deadzone_x2 = min(smoothed_cx + deadzone_width // 2, canvas_width)
            deadzone_y2 = min(smoothed_cy + deadzone_height // 2, canvas_height)

            if conf_parameters["show_tracker"]:
                # Draw face box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw dead zone box
                cv2.rectangle(frame, (deadzone_x1, deadzone_y1), (deadzone_x2, deadzone_y2), (255, 0, 255), 2)

            

            # Update only if face outside dead zone
            if cx < deadzone_x1 or cx > deadzone_x2:
                smoothed_cx = int(smoothed_cx * (1 - conf_parameters["smoothing_factor"]) + cx * conf_parameters["smoothing_factor"])
            if cy < deadzone_y1 or cy > deadzone_y2:
                smoothed_cy = int(smoothed_cy * (1 - conf_parameters["smoothing_factor"]) + cy * conf_parameters["smoothing_factor"])

    elif tracking_enabled:
        face_visible = False

    # === Zoom logic ===
    show_zoomed = tracking_enabled and (
        (face_visible or (current_time - last_seen_time < conf_parameters["face_lost_timeout"]))
        and smoothed_cx is not None
    )


    # === Zoom Logic with Smooth Transition ===

    # Compute new crop box if zoomed
    if show_zoomed:
        zoom_width = int(canvas_width / conf_parameters["zoom_factor"])
        zoom_height = int(canvas_height / conf_parameters["zoom_factor"])
        center_x = smoothed_cx + manual_offset_x
        center_y = smoothed_cy + manual_offset_y

        x_start = max(min(center_x - zoom_width // 2, canvas_width - zoom_width), 0)
        y_start = max(min(center_y - zoom_height // 2, canvas_height - zoom_height), 0)
        x_end = x_start + zoom_width
        y_end = y_start + zoom_height

        target_crop_box = (x_start, y_start, x_end, y_end)
    else:
        target_crop_box = (0, 0, canvas_width, canvas_height)
        smoothed_cx, smoothed_cy = None, None
        manual_offset_x, manual_offset_y = 0, 0

    # Start a new transition if the target changes
    if current_crop_box != target_crop_box:
        prev_crop_box = current_crop_box if current_crop_box else target_crop_box
        current_crop_box = target_crop_box
        transition_counter = transition_frames

    # Interpolate between previous and target crop boxes
    if transition_counter > 0 and prev_crop_box:
        t = 1 - (transition_counter / transition_frames)
        x1 = lerp(prev_crop_box[0], current_crop_box[0], t)
        y1 = lerp(prev_crop_box[1], current_crop_box[1], t)
        x2 = lerp(prev_crop_box[2], current_crop_box[2], t)
        y2 = lerp(prev_crop_box[3], current_crop_box[3], t)
        transition_counter -= 1
    else:
        x1, y1, x2, y2 = current_crop_box

    # Extract and resize zoom region
    cropped_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped_frame, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)


    desired_width = 1920
    desired_height = 1080

    # Assuming zoomed_frame is your final frame to send
    resized_frame = cv2.resize(zoomed_frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

    # Then convert to BGRA for NDI
    bgra_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2BGRA)

    video_frame.xres = desired_width
    video_frame.yres = desired_height
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    video_frame.line_stride_in_bytes = desired_width * 4
    video_frame.data = bgra_frame
    ndi.send_send_video_v2(ndi_sender, video_frame)



    
    # === Draw HUD (🎯 HUD stands for Heads-Up Display) ===

    preview_frame = zoomed_frame.copy()
    hud_color = (0, 255, 0) if tracking_enabled else (0, 0, 255)
    HUD_Y_STARTING_POSITION = 30
    HUD_X_STARTING_POSITION = 10

    if conf_parameters["show_hud"]:
        HUD_Display(preview_frame, f"Tracking: {'ON' if tracking_enabled else 'OFF'}", HUD_X_STARTING_POSITION, HUD_Y_STARTING_POSITION, hud_color)
        for config in conf_parameters:
            HUD_Y_STARTING_POSITION += 30
            HUD_Display(preview_frame, f"{config}: {conf_parameters[config]}", HUD_X_STARTING_POSITION, HUD_Y_STARTING_POSITION, hud_color)

        if not face_visible and (current_time - last_seen_time > conf_parameters["face_lost_timeout"]):
            HUD_Display(preview_frame, "🔍Face lost", HUD_X_STARTING_POSITION, HUD_Y_STARTING_POSITION + 30, (0, 0, 255))


    cv2.imshow(WINDOW_NAME, preview_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
        print("▶️ Tracking" if tracking_enabled else "🛑 Tracking suspended")

    elif key == ord('f'):
        conf_parameters["show_tracker"] = not conf_parameters["show_tracker"]
        print("▶️ Face Tracker Active" if conf_parameters["show_tracker"] else "🛑 Face Tracker Inactive")

    elif key == ord('h'):
        conf_parameters["show_hud"] = not conf_parameters["show_hud"]
        print("▶️ HUD Active" if conf_parameters["show_tracker"] else "🛑 HUD Inactive")


    elif key == ord('+'):
        conf_parameters["zoom_factor"] = min(conf_parameters["zoom_factor"] + 0.1, 3.0)
        print(f"Zoom Factor: {conf_parameters['zoom_factor']:.1f}")

    elif key == ord('-'):
        conf_parameters["zoom_factor"] = max(conf_parameters["zoom_factor"] - 0.1, 1.1)
        print(f"Zoom Factor: {conf_parameters['zoom_factor']:.1f}")



    elif key == ord('n'):
        conf_parameters["dead_zone_scale"] = min(conf_parameters["dead_zone_scale"] + 1.0, 10.0)
        print(f"Deadzone_scale: {conf_parameters['dead_zone_scale']:.1f}")

    elif key == ord('m'):
        conf_parameters["dead_zone_scale"] = max(conf_parameters["dead_zone_scale"] - 1.0, 1.2)
        print(f"Deadzone_scale: {conf_parameters['dead_zone_scale']:.1f}")

    
    elif key == ord('j'):
        conf_parameters["smoothing_factor"] = min(conf_parameters["smoothing_factor"] + 0.01, 0.2)
        print(f"Smoothing_factor: {conf_parameters['smoothing_factor']:.2f}")

    elif key == ord('k'):
        conf_parameters["smoothing_factor"] = max(conf_parameters["smoothing_factor"] - 0.01, 0.05)
        print(f"Smoothing_factor: {conf_parameters['smoothing_factor']:.2f}")



cap.release()
cv2.destroyAllWindows()

ndi.send_destroy(ndi_sender)
ndi.destroy()

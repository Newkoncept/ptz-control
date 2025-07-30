import cv2

# ---- Settings ----
VIDEO_SOURCE = 16  # Change to 0, 1, or 2 depending on your camera
WINDOW_TITLE = "Live Face Detection (Always on Top)"

# ---- Initialize Video Feed ----
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Cannot open video source")
    exit()

# ---- Determine Resolution ----
ret, img = cap.read()
if not ret:
    print("❌ Error: Cannot read first frame")
    cap.release()
    exit()

# Optional resize logic
if img.shape[1] / img.shape[0] > 1.55:
    res = (256, 144)
else:
    res = (216, 162)

# ---- Load Haar Cascade ----
cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")


# ---- Main Loop ----
print("✅ Running... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Failed to grab frame")
        break

    # resized = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = frame

    # Face Detection
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(WINDOW_TITLE, resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()
print("🛑 Stopped")

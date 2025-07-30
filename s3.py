import cv2
import numpy as np

WINDOW_TITLE = "Dynamic Zoom on Face"
VIDEO_SOURCE = 16  # Change to 0 or 1 depending on your setup

# Load Haar cascade
cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Open camera
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open video source.")
    exit()

print("🎥 Starting... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Convert frame to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with the detected face(s)
    cv2.imshow(WINDOW_TITLE, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os

# Folder to save your images
save_path = "faces_db/Me"
os.makedirs(save_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_count = 31
MAX_IMAGES = 60  # Number of images to capture

print("📸 Collecting face images... Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face
        face_crop = frame[y:y + h, x:x + w]

        # Save face image
        if img_count < MAX_IMAGES:
            img_name = os.path.join(save_path, f"{img_count + 1}.jpg")
            cv2.imwrite(img_name, face_crop)
            img_count += 1
            print(f"Saved {img_name}")

    cv2.imshow("Face Collector", frame)

    # Exit if 'q' is pressed or enough images are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Collected {img_count} images in {save_path}")

import cv2
import threading
from deepface import DeepFace

REFERENCE_IMG = "faces_db/kanak.jpg"
ref_img = cv2.imread(REFERENCE_IMG)

result_text = "Loading..."
lock = threading.Lock()

def verify_face(frame):
    global result_text
    try:
        result = DeepFace.verify(frame, ref_img, enforce_detection=False, model_name="Facenet")
        with lock:
            result_text = "Matched" if result["verified"] else "Unmatched"
    except:
        with lock:
            result_text = "No Face Detected"

def background_verification(frame):
    thread = threading.Thread(target=verify_face, args=(frame,))
    thread.daemon = True
    thread.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start verification in background every 10 frames
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
        background_verification(frame.copy())

    # Display result
    with lock:
        cv2.putText(frame, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

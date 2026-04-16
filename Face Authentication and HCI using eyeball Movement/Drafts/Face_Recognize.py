import cv2
import numpy as np
from deepface import DeepFace
import joblib

# Load your stored embeddings
my_embeddings = joblib.load("my_face_embeddings.pkl")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cap = cv2.VideoCapture(0)
print("🎥 Starting recognition... Press 'q' to quit.")
fc=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fc+=1
    if fc%5!=0:
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get embedding for current frame
            embedding = DeepFace.represent(rgb_frame, model_name="Facenet512", enforce_detection=False)[0]["embedding"]

            # Compare with your embeddings
            sims = [cosine_similarity(embedding, e) for e in my_embeddings]
            best_sim = max(sims)

            if best_sim > 0.8:  # Threshold (tune as needed)
                text, color = "It's Me!", (0, 255, 0)
                print(text,fc)
                break
            else:
                text, color = "Not Me", (0, 0, 255)
                print(text,fc);
            fc=0

        except:
            text, color = "No Face Detected", (0, 0, 255)

        # Display result
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print("You are Verified")
cap.release()
cv2.destroyAllWindows()
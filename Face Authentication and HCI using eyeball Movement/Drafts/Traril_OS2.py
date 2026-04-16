import cv2
import mediapipe as mp
import pyautogui as pg
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from deepface import DeepFace
import joblib

pg.FAILSAFE = False

# --- MediaPipe setup ---
facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Audio setup ---
devices = AudioUtilities.GetSpeakers() 
interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
step_vol = 0.05

# --- Screen info ---
scw, sch = pg.size()
prev_x, prev_y = 0, 0
lastbt = 0
blink_count = 0
Sn = 20.0  # Sensitivity factor

# --- Load embeddings for face verification ---
my_embeddings = joblib.load("my_face_embeddings.pkl")

# --- Cosine similarity function ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Start VideoCapture ---
cap = cv2.VideoCapture(0)
mode = "verification"  # Start with face verification
fc = 0  # Frame counter to skip some frames for efficiency
text=""
print("🎥 Starting system... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameh, framew, _ = frame.shape
    fc += 1

    if mode == "verification" and fc % 5 == 0:  # Skip some frames to reduce lag
        try:
            embedding = DeepFace.represent(rgb_frame, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            sims = [cosine_similarity(embedding, e) for e in my_embeddings]
            best_sim = max(sims)

            if best_sim > 0.8:
                mode = "cursor_control"  # Switch mode
                print("✅ User Verified! Switching to Cursor Control")
            else:
                text = "Unknown"
        except:
            text = "No Face Detected"

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    elif mode == "cursor_control":
        # --- Face mesh for cursor control ---
        output = facemesh.process(rgb_frame)
        result = hands.process(rgb_frame)
        lanP = output.multi_face_landmarks

        if lanP:
            lan = lanP[0].landmark
            head_x = lan[1].x
            head_y = lan[1].y
            scx = int((head_x - 0.5) * scw * Sn + scw / 2)
            scy = int((head_y - 0.5) * sch * Sn + sch / 2)
            prev_x = (prev_x * 0.8) + (scx * 0.2)
            prev_y = (prev_y * 0.8) + (scy * 0.2)
            pg.moveTo(prev_x, prev_y, duration=0.1)

            # --- Blink detection ---
            left_eye = [lan[145], lan[159]]
            if abs(left_eye[0].y - left_eye[1].y) < 0.004:
                current_time = time.time()
                if current_time - lastbt < 0.3:
                    blink_count += 1
                else:
                    blink_count = 1
                lastbt = current_time

                if blink_count == 2:
                    pg.doubleClick()
                    print("Double Click Detected!")
                    blink_count = 0
                else:
                    pg.click()
                    print("Left Click Detected!")
                time.sleep(0.1)

            right_eye = [lan[374], lan[386]]
            if abs(right_eye[0].y - right_eye[1].y) < 0.004:
                pg.rightClick()
                print("Right Click Detected!")
                time.sleep(0.1)

            # Draw eyes
            for lans in left_eye + right_eye:
                x = int(lans.x * framew)
                y = int(lans.y * frameh)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # --- Hand gestures for volume ---
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                thumb = hl.landmark[4]
                index = hl.landmark[8]
                middle = hl.landmark[12]

                thumb_index_distance = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)
                thumb_middle_distance = np.sqrt((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2)

                if thumb_index_distance < 0.05:
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    new_volume = max(0.0, current_volume - step_vol)
                    volume.SetMasterVolumeLevelScalar(new_volume, None)
                    print("Volume Decreased")
                elif thumb_middle_distance < 0.05:
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    new_volume = min(1.0, current_volume + step_vol)
                    volume.SetMasterVolumeLevelScalar(new_volume, None)
                    print("Volume Increased")

        # Display volume bar
        current_volume = volume.GetMasterVolumeLevelScalar()
        cv2.putText(frame, f"Volume: {int(current_volume * 100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 70), (300, 100), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 70), (50 + int(current_volume * 250), 100), (0, 255, 0), -1)

        cv2.putText(frame, "Cursor Control Active", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Show frame ---
    cv2.imshow("Face & Cursor Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if mode == "cursor_control":
            mode = "verification"  # Return to verification
            print("🔄 Returning to Face Verification...")
        else:
            break  # Quit program

cap.release()
cv2.destroyAllWindows()

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import pyautogui as pg
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import os
from deepface import DeepFace
import joblib
import threading

faces_folder = "faces_db/Me"
os.makedirs(faces_folder, exist_ok=True)
temp_embeddings_file = "my_face_embeddings.pkl"

if os.path.exists(temp_embeddings_file):
    embeddings = joblib.load(temp_embeddings_file)
else:
    embeddings = np.array([])


cap = cv2.VideoCapture(0)
pg.FAILSAFE = False

# MediaPipe Face Mesh and Hands
facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
step_vol = 0.05

# Screen info
scw, sch = pg.size()
prev_x, prev_y = 0, 0
lastbt = 0
blink_count = 0
Sn = 20.0
mouse_running = False

# Status string and color for overlay
camera_status = "Verifying..."
camera_status_color = (255, 255, 0)

# ---------------- Helper Functions ----------------
def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Threaded decorator
def threaded(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper

# ---------------- Camera Update Loop ----------------
def update_frame():
    global prev_x, prev_y, lastbt, blink_count, mouse_running, camera_status, camera_status_color
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = cv2.flip(frame, 1)
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameh, framew, _ = frame.shape

    # Face mesh
    face_landmarks = facemesh.process(rgbframe).multi_face_landmarks

    # During verification, draw face rectangle
    if face_landmarks and not mouse_running and camera_status in ["Verifying...", "Unknown User"]:
        x_min = int(min([l.x for l in face_landmarks[0].landmark])*framew)
        x_max = int(max([l.x for l in face_landmarks[0].landmark])*framew)
        y_min = int(min([l.y for l in face_landmarks[0].landmark])*frameh)
        y_max = int(max([l.y for l in face_landmarks[0].landmark])*frameh)
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),2)

    if face_landmarks:
        lan = face_landmarks[0].landmark

        # Draw only eyes for reference
        eyes_ids = [145,159,374,386]
        for idx in eyes_ids:
            x = int(lan[idx].x * framew)
            y = int(lan[idx].y * frameh)
            cv2.circle(frame,(x,y),3,(0,0,255),-1)

        if mouse_running:
            # Head-based cursor: use midpoint between eyes
            lx = (lan[145].x + lan[159].x)/2
            ly = (lan[145].y + lan[159].y)/2
            scx = int((lx - 0.5) * scw * Sn + scw / 2)
            scy = int((ly - 0.5) * sch * Sn + sch / 2)
            prev_x = (prev_x * 0.8) + (scx * 0.2)
            prev_y = (prev_y * 0.8) + (scy * 0.2)
            pg.moveTo(prev_x, prev_y, duration=0.1)

            # Eye blink clicks
            left_eye = [lan[145], lan[159]]
            right_eye = [lan[374], lan[386]]
            current_time = time.time()

            if abs(left_eye[0].y - left_eye[1].y) < 0.004:
                if current_time - lastbt < 0.3:
                    blink_count += 1
                else:
                    blink_count = 1
                lastbt = current_time
                if blink_count == 2:
                    pg.doubleClick()
                    blink_count = 0
                else:
                    pg.click()
                time.sleep(0.1)
            if abs(right_eye[0].y - right_eye[1].y) < 0.004:
                pg.rightClick()
                time.sleep(0.1)

    # Hand gestures for volume
    if mouse_running:
        result = hands.process(rgbframe)
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                thumb = hl.landmark[4]
                index = hl.landmark[8]
                middle = hl.landmark[12]
                current_volume = volume.GetMasterVolumeLevelScalar()
                if np.linalg.norm([thumb.x-index.x, thumb.y-index.y]) < 0.05:
                    volume.SetMasterVolumeLevelScalar(max(0.0, current_volume-step_vol), None)
                elif np.linalg.norm([thumb.x-middle.x, thumb.y-middle.y]) < 0.05:
                    volume.SetMasterVolumeLevelScalar(min(1.0, current_volume+step_vol), None)
                current_volume = volume.GetMasterVolumeLevelScalar()
                cv2.putText(frame,f"Volume: {int(current_volume*100)}%",(50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    # Overlay status
    cv2.putText(frame, camera_status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, camera_status_color, 2)

    # Convert for Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(15, update_frame)

# ---------------- Core Functions ----------------
@threaded
def verify_user(auto_start=False):
    global camera_status, camera_status_color, mouse_running
    camera_status = "Verifying..."
    camera_status_color = (255,255,0)
    status_label.config(text="🔍 Verifying User...")

    verified = False
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            embedding = DeepFace.represent(rgb_frame, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            sims = [cosine_similarity(embedding,e) for e in embeddings]
            best_sim = max(sims) if len(sims)>0 else 0
            if best_sim > 0.8:
                verified = True
                break
        except:
            continue
        time.sleep(0.05)

    if verified:
        camera_status = "Verified User"
        camera_status_color = (0,255,0)
        status_label.config(text="✅ User Verified")
        if auto_start:
            StartMouse()
    else:
        camera_status = "Unknown User"
        camera_status_color = (0,0,255)
        status_label.config(text="❌ Unknown User")

# ---------------- Add / Delete / Mouse ----------------
add_count = 0
MAX_ADD = 20  # Limit per session

@threaded
def add_face():
    global embeddings, camera_status, camera_status_color, add_count
    ret, frame = cap.read()
    if not ret:
        return
    frameh, framew, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    facemesh_results = facemesh.process(rgb_frame).multi_face_landmarks
    if facemesh_results:
        lan = facemesh_results[0].landmark
        # Midpoint between eyes to crop
        cx = int((lan[145].x + lan[159].x)/2 * framew)
        cy = int((lan[145].y + lan[159].y)/2 * frameh)
        size = max(framew, frameh)//4
        x, y = max(cx-size//2,0), max(cy-size//2,0)
        face_crop = frame[y:y+size, x:x+size]
        # Save
        img_path = os.path.join(faces_folder,f"{len(os.listdir(faces_folder))+1}.jpg")
        cv2.imwrite(img_path, face_crop)
        # Embedding
        try:
            emb = DeepFace.represent(face_crop, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            if embeddings.size>0 and max([cosine_similarity(emb,e) for e in embeddings])>0.8:
                camera_status="User Already Registered"
                camera_status_color=(0,0,255)
                status_label.config(text="⚠️ User Already Registered")
                return
            embeddings = np.append(embeddings,[emb],axis=0) if embeddings.size else np.array([emb])
        except:
            pass
        add_count +=1
        camera_status=f"Adding Face {add_count}/{MAX_ADD}"
        camera_status_color=(0,255,0)
        joblib.dump(embeddings,temp_embeddings_file)
        if add_count>=MAX_ADD:
            camera_status="Face Adding Complete"
            camera_status_color=(0,255,0)
            add_count=0
            status_label.config(text="✅ Face Adding Complete")

def delete_faces():
    global embeddings, camera_status, camera_status_color
    for f in os.listdir(faces_folder):
        os.remove(os.path.join(faces_folder,f))
    embeddings=np.array([])
    if os.path.exists(temp_embeddings_file):
        os.remove(temp_embeddings_file)
    camera_status="Home Mode"
    camera_status_color=(0,255,0)
    status_label.config(text="🗑️ Temporary embeddings deleted")

def StartMouse():
    global mouse_running, camera_status, camera_status_color
    mouse_running=True
    camera_status="Mouse Control Active"
    camera_status_color=(255,255,0)
    status_label.config(text="✅ Mouse Control Started")

def StopMouse():
    global mouse_running, camera_status, camera_status_color
    mouse_running=False
    camera_status="Home Mode"
    camera_status_color=(0,255,0)
    status_label.config(text="✅ Home Mode")

# ---------------- Tkinter GUI ----------------
root = tk.Tk()
root.title("Face Recognition GUI")
video_label = tk.Label(root)
video_label.pack()
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame,text="Verify Face",width=15,command=lambda:verify_user(auto_start=True)).grid(row=0,column=0,padx=5)
tk.Button(btn_frame,text="Add Face",width=15,command=add_face).grid(row=0,column=1,padx=5)
tk.Button(btn_frame,text="Delete Faces",width=15,command=delete_faces).grid(row=0,column=2,padx=5)
tk.Button(btn_frame,text="Start Mouse",width=15,command=StartMouse).grid(row=0,column=3,padx=5)
tk.Button(btn_frame,text="Stop Mouse",width=15,command=StopMouse).grid(row=0,column=4,padx=5)
status_label=tk.Label(root,text="Status: Waiting...",font=("Arial",12),fg="blue")
status_label.pack(pady=10)

update_frame()
verify_user(auto_start=True)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
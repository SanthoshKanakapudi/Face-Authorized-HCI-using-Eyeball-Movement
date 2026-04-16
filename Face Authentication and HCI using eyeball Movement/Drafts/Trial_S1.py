import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import pyautogui as pg
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
cap = cv2.VideoCapture(0)

pg.FAILSAFE=False
# cap = cv2.VideoCapture(0)
facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, 0, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
scw, sch = pg.size()
prev_x, prev_y = 0, 0
lastbt = 0
blink_count = 0

Sn = 20.0  # Sensitivity factor
step_vol = 0.05
mouse_running = False  # control flag

def update_frame():
    global prev_x, prev_y, lastbt, blink_count, mouse_running
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = cv2.flip(frame, 1)
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mouse_running:
        output = facemesh.process(rgbframe)
        result = hands.process(rgbframe)

        lanP = output.multi_face_landmarks
        frameh, framew, _ = frame.shape

        if lanP:
            lan = lanP[0].landmark
            # Move mouse with head
            head_x, head_y = lan[1].x, lan[1].y
            scx = int((head_x - 0.5) * scw * Sn + scw / 2)
            scy = int((head_y - 0.5) * sch * Sn + sch / 2)
            prev_x = (prev_x * 0.8) + (scx * 0.2)
            prev_y = (prev_y * 0.8) + (scy * 0.2)
            pg.moveTo(prev_x, prev_y, duration=0.1)

            # Eye blinks for click
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
                    blink_count = 0
                else:
                    pg.click()
                time.sleep(0.1)

            right_eye = [lan[374], lan[386]]
            if abs(right_eye[0].y - right_eye[1].y) < 0.004:
                pg.rightClick()
                time.sleep(0.1)

            # Draw eyes
            for lans in left_eye + right_eye:
                x, y = int(lans.x * framew), int(lans.y * frameh)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Hand gestures for volume
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                thumb = hl.landmark[4]
                index = hl.landmark[8]
                middle = hl.landmark[12]

                thumb_index_distance = np.linalg.norm(
                    [thumb.x - index.x, thumb.y - index.y]
                )
                thumb_middle_distance = np.linalg.norm(
                    [thumb.x - middle.x, thumb.y - middle.y]
                )

                if thumb_index_distance < 0.05:
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - step_vol), None)
                elif thumb_middle_distance < 0.05:
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + step_vol), None)

                current_volume = volume.GetMasterVolumeLevelScalar()
                cv2.putText(frame, f"Volume: {int(current_volume * 100)}%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Home Mode", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      
    # Convert OpenCV BGR → RGB
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Repeat after 15 ms
    root.after(15, update_frame)

# Button functions
def Home():
    print("You're in Home")
    status_label.config(text="✅ You're in Home...")

def add_face():
    print("Add Face pressed")
    status_label.config(text="✅ Adding Face...")

def delete_face():
    print("Delete Face pressed")
    status_label.config(text="🗑️ Deleting Face...")

def verify_face():
    print("Verifying Face....")
    status_label.config(text="🔍Verifying Face....")

def StartMouse():
    print("Started Mouse Control")
    global mouse_running
    mouse_running = True
    status_label.config(text="Starting Mouse Control")

def StopMouse():
    print("Stopped Mouse Control")
    global mouse_running
    mouse_running = False
    status_label.config(text="✅ Home Mode...")



# Tkinter window
root = tk.Tk()
root.title("Face Recognition UI")

# Camera feed label
video_label = tk.Label(root)
video_label.pack()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Home", width=15, command=Home).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Add Face", width=15, command=add_face).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Delete Face", width=15, command=delete_face).grid(row=0, column=2, padx=5)
tk.Button(btn_frame, text="Verify Face", width=15, command=verify_face).grid(row=0, column=3, padx=5)
tk.Button(btn_frame, text="Start", width=15, command=StartMouse).grid(row=0, column=4, padx=5)
tk.Button(btn_frame, text="Stop", width=15, command=StopMouse).grid(row=0, column=5, padx=5)

# Status Label
status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)
# Start video loop
update_frame()

# Run the app
root.mainloop()

# Release camera when closed
cap.release()
cv2.destroyAllWindows()
# Face Recognition Based Hands-Free Computer Control System

Overview
  This project is a computer vision–based human-computer interaction system that allows users to control the mouse and system volume using face recognition, eye movements, and hand gestures.
The system first verifies the user using facial recognition. Once verified, it enables hands-free control of the mouse pointer and system functions.

#🚀 Features

1. Face Verification (Security Layer)  
  Uses deep learning (FaceNet via DeepFace) to authenticate users and prevent unauthorized access  

2. Head-Based Mouse Control  
  Cursor moves based on head/eye position with smooth tracking  

3. Eye Blink Detection for Clicks  
  - Left eye blink → Left click  
  - Double blink → Double click  
  - Right eye blink → Right click  

4. Hand Gesture Volume Control  
  - Thumb + Index finger → Decrease volume  
  - Thumb + Middle finger → Increase volume  

5.  Face Data Management  
  - Add new face embeddings  
  - Delete stored faces  
  - Prevent duplicate registrations  

6. Real-Time GUI  
  Built using Tkinter with live camera feed and status display  

#Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- DeepFace  
- Tkinter  
- PyAutoGUI  
- Pycaw  
- NumPy  
- Joblib  

#How It Works

1. Camera captures real-time video  
2. Face is verified using stored embeddings  
3. If verified:  
   - Head movement controls cursor  
   - Eye blinks trigger mouse clicks  
   - Hand gestures adjust system volume  
4. If not verified: system remains locked  

#Project Structure

faces_db/Me/            # Stored face images  
my_face_embeddings.pkl  # Saved face embeddings  
main.py                 # Main application file  

#Usage

1. Run the program  
2. Click "Add Face" to register your face  
3. Click "Verify Face"  
4. Once verified:  
   - Move head → Move cursor  
   - Blink → Click  
   - Use hand gestures → Control volume
     
## ⚠️ Limitations

- Requires good lighting conditions  
- Accuracy may vary with camera quality right now but can be modified...
- Not suitable for critical security applications  
- Performance depends on system hardware  


# Future Improvements

- Add voice control  
- Improve multi-user support  
- Add gesture-based scrolling  
- Optimize performance for low-end systems  

#License

This project is for educational and research purposes only.

import os
import cv2
from deepface import DeepFace

# Disable oneDNN optimizations to avoid numerical differences or warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def analyze_faces(frame):
    # Analyze the frame for facial attributes
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
    for face_data in results:
        # Extract attributes
        age = face_data['age']
        gender = face_data['gender']
        emotion = max(face_data['emotion'], key=face_data['emotion'].get)

        # Draw information on the frame
        x, y, w, h = face_data['region']['x'], face_data['region']['y'], face_data['region']['w'], face_data['region']['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Access the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the frame
        processed_frame = analyze_faces(frame)

        # Display the frame
        cv2.imshow('Real-Time Face Analysis', processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

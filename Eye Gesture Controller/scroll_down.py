import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
from pynput.keyboard import Controller

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points, landmarks, img_w, img_h):
    p = [landmarks[pt] for pt in eye_points]
    coords = [(int(pn.x * img_w), int(pn.y * img_h)) for pn in p]
    vertical1 = math.dist(coords[1], coords[5])
    vertical2 = math.dist(coords[2], coords[4])
    horizontal = math.dist(coords[0], coords[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)

blink_threshold = 0.21
blink_counter = 0
blink_state = False

keyboard = Controller()  # Initialize keyboard controller

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(LEFT_EYE, mesh, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, mesh, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < blink_threshold:
            if not blink_state:
                blink_counter += 1
                blink_state = True

                # Trigger actions on blink
                print("Blink detected!")
                
                # Action 1: Scroll down the mouse
                pyautogui.scroll(-10)
                
                
        else:
            blink_state = False

        # Show EAR on screen
        cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Blinks: {blink_counter}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 100), 2)

    cv2.imshow("Eye Blink Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

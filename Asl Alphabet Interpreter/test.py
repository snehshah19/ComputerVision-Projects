import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
import pickle
from collections import deque, Counter
import time

# Load model and label encoder
model = load_model("asl_mediapipe_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# TTS setup
engine = pyttsx3.init()

# Prediction smoothing
predictions_queue = deque(maxlen=15)
prev_prediction = ''
text_buffer = ''
last_prediction_time = time.time()
PREDICTION_INTERVAL = 0.5

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and flatten landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if time.time() - last_prediction_time > PREDICTION_INTERVAL:
                prediction = model.predict(np.expand_dims(landmarks, axis=0))[0]
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                predictions_queue.append(predicted_label)
                last_prediction_time = time.time()

    # Stable prediction
    if len(predictions_queue) == predictions_queue.maxlen:
        most_common = Counter(predictions_queue).most_common(1)[0]
        if most_common[1] > 10:
            stable_prediction = most_common[0]
        else:
            stable_prediction = None
    else:
        stable_prediction = None

    # Display prediction and text buffer
    if stable_prediction:
        cv2.putText(frame, stable_prediction, (160, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        if stable_prediction != prev_prediction:
            text_buffer += stable_prediction
            engine.say(stable_prediction)
            engine.runAndWait()
            prev_prediction = stable_prediction
    else:
        cv2.putText(frame, "Detecting...", (160, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.putText(frame, text_buffer, (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("ASL Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
from difflib import get_close_matches

# Load model and label encoder
model = tf.keras.models.load_model("asl_mediapipe_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load English words for suggestions
with open("words.txt") as f:
    word_list = f.read().split()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Text-to-Speech engine
engine = pyttsx3.init()

# Normalize function (if required)
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)
    base = landmarks[0]
    normalized = landmarks - base
    return normalized.flatten()

# Main
cap = cv2.VideoCapture(0)
current_prediction = ""
current_word = ""
sentence = ""
suggestion_list = []

print("Press 'a' to add letter to word, 'c' to clear, 's' to speak the word, 'q' to quit.")
print("Press '1', '2', '3' to select a suggestion, 'Enter' to finalize the word, 'Backspace' to remove word")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                normalized = normalize_landmarks(landmarks).reshape(1, -1)
                pred = model.predict(normalized)
                class_id = np.argmax(pred)
                current_prediction = le.inverse_transform([class_id])[0]

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and current_prediction:
        current_word += current_prediction
        suggestion_list = get_close_matches(current_word.lower(), word_list, n=3)
    elif key == ord('c'):
        current_word = ""
        suggestion_list = []
    elif key == ord('s') and current_word:
        engine.say(current_word)
        engine.runAndWait()
    elif key == ord('q'):
        break
    elif key == ord('1') and suggestion_list:
        current_word = suggestion_list[0]  # Auto-complete using the first suggestion
        suggestion_list = []
    elif key == ord('2') and len(suggestion_list) > 1:
        current_word = suggestion_list[1]  # Select the second suggestion
        suggestion_list = []
    elif key == ord('3') and len(suggestion_list) > 2:
        current_word = suggestion_list[2]  # Select the third suggestion
        suggestion_list = []
    elif key == 13:  # Enter key
        sentence += current_word + " "
        current_word = ""
        suggestion_list = []
    elif key == 8:  # Backspace key
        sentence = sentence.rstrip()  # Remove last word
        current_word = ""
        suggestion_list = []

    # Display info
    cv2.putText(frame, f"Letter: {current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for i, sug in enumerate(suggestion_list):
        cv2.putText(frame, f"Suggestion {i+1}: {sug}", (10, 150 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("ASL Interpreter", frame)

cap.release()
cv2.destroyAllWindows()



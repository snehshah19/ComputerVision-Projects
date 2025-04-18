# collect_landmarks.py
import cv2
import mediapipe as mp
import csv
import os
import time

# Create directory if not exists
os.makedirs("data", exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Letters A-Z
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_label_index = 0
samples_per_class = 200
data = []
collected = 0

cap = cv2.VideoCapture(0)
print(f"Starting collection for: {labels[current_label_index]}")

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

            # Extract landmarks as flat list: [x1, y1, z1, ..., x21, y21, z21]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == 63:
                data.append([labels[current_label_index]] + landmarks)
                collected += 1
                time.sleep(0.05)  # slight delay for sampling
            
            cv2.putText(frame, f"Collecting: {labels[current_label_index]} ({collected}/{samples_per_class})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)
    if key == ord('n'):
        if collected >= samples_per_class:
            current_label_index += 1
            collected = 0
            print(f"Next letter: {labels[current_label_index] if current_label_index < len(labels) else 'Done'}")
        else:
            print(f"Need at least {samples_per_class} samples. Currently have {collected}.")
    if key == ord('q') or current_label_index >= len(labels):
        break

# Save to CSV
with open("data/asl_landmark_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    writer.writerow(header)
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
import cv2
import mediapipe as mp
import random
import time

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game options
options = ['rock', 'paper', 'scissors']

# Finger tip landmarks
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky

# Helper function to detect gesture
def detect_gesture(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = fingers.count(1)

    # Decide gesture
    if total_fingers == 0:
        return 'rock'
    elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
        return 'scissors'
    elif total_fingers == 5:
        return 'paper'
    else:
        return 'unknown'

# Start capturing
cap = cv2.VideoCapture(0)
game_started = False
start_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, f'Your move: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if gesture in options and not game_started:
                game_started = True
                start_time = time.time()
                user_move = gesture
                comp_move = random.choice(options)

    if game_started:
        if time.time() - start_time > 2:  # wait 2 seconds then show result
            cv2.putText(frame, f'Computer: {comp_move}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if user_move == comp_move:
                result_text = "Draw!"
            elif (user_move == 'rock' and comp_move == 'scissors') or \
                 (user_move == 'paper' and comp_move == 'rock') or \
                 (user_move == 'scissors' and comp_move == 'paper'):
                result_text = "You Win!"
            else:
                result_text = "You Lose!"
            cv2.putText(frame, result_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

            # reset game
            if time.time() - start_time > 5:
                game_started = False

    cv2.imshow("Rock Paper Scissors", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

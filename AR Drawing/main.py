import cv2
import numpy as np
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Drawing variables
canvas = None
draw_color = (255, 0, 255)
brush_thickness = 5
eraser_thickness = 30
tool = "brush"
prev_x, prev_y = 0, 0

# Finger tip indices (for detection)
finger_tips = [8, 12, 16, 20]

# Color/Tool buttons
color_buttons = {
    "pink": ((40, 40), (140, 140), (255, 0, 255)),
    "blue": ((160, 40), (260, 140), (255, 0, 0)),
    "green": ((280, 40), (380, 140), (0, 255, 0)),
    "eraser": ((400, 40), (500, 140), (0, 0, 0))
}

# Gesture delay lock (to avoid multiple triggers)
last_clear_time = 0
clear_cooldown = 2  # seconds

def fingers_up(hand_landmarks):
    fingers = []
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Start loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Draw UI toolbar
    for name, ((x1, y1), (x2, y2), color) in color_buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1+5, y2+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Detect hand
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            fingers = fingers_up(hand_landmarks)
            total_fingers = sum(fingers)

            # Feature 1: Clear canvas with 5 fingers up
            if total_fingers == 4 and (time.time() - last_clear_time) > clear_cooldown:
                canvas = np.zeros_like(frame)
                last_clear_time = time.time()
                cv2.putText(frame, "Canvas Cleared!", (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Feature 2: Toolbar interaction
            if y < 140:
                for name, ((x1, y1), (x2, y2), color) in color_buttons.items():
                    if x1 < x < x2 and y1 < y < y2:
                        if name == "eraser":
                            tool = "eraser"
                        else:
                            draw_color = color
                            tool = "brush"

            # Drawing mode - only index finger up
            elif fingers[0] == 1 and fingers[1:] == [0, 0, 0]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                if tool == "brush":
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                elif tool == "eraser":
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), eraser_thickness)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

    # Overlay canvas
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("AR Drawing App", frame)

    # Feature 3: Save with 's'
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved as {filename}")

cap.release()
cv2.destroyAllWindows()

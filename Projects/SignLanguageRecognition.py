import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def is_finger_up(tip, pip):
    return tip.y < pip.y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No Gesture"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip  = lm[mp_hands.HandLandmark.THUMB_IP]

            index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]

            pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]

            fingers_up = [
                is_finger_up(index_tip, index_pip),
                is_finger_up(middle_tip, middle_pip),
                is_finger_up(ring_tip, ring_pip),
                is_finger_up(pinky_tip, pinky_pip)
            ]

            if all(fingers_up):
                gesture_text = "HELLO (Open Palm)"

            elif fingers_up == [False, False, False, False]:
                gesture_text = "FIST"

            elif fingers_up == [True, False, False, False]:
                gesture_text = "ONE"

            elif fingers_up == [True, True, False, False]:
                gesture_text = "TWO / PEACE"

            elif fingers_up == [True, True, True, False]:
                gesture_text = "THREE"

            elif fingers_up == [True, True, True, True]:
                gesture_text = "FOUR"

            elif (not fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3] 
                  and thumb_tip.x > thumb_ip.x):
                gesture_text = "THUMBS UP"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (10, 10), (360, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {gesture_text}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
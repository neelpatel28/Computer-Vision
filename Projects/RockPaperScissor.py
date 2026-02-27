import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

user_score = 0
computer_score = 0
choices = ["Rock", "Paper", "Scissors"]

round_active = False
countdown_start = None
computer_choice = None
round_result = ""

def detect_gesture(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []
    for tip in tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    if fingers == [False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True]:
        return "Paper"
    elif fingers == [True, True, False, False]:
        return "Scissors"
    return None

def winner(user, comp):
    if user == comp:
        return "Tie"
    if (user == "Rock" and comp == "Scissors") or \
       (user == "Paper" and comp == "Rock") or \
       (user == "Scissors" and comp == "Paper"):
        return "User"
    return "Computer"

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    user_move = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            user_move = detect_gesture(hand_landmarks)

    if not round_active and user_move:
        round_active = True
        countdown_start = time.time()
        round_result = ""
        computer_choice = None

    if round_active:
        elapsed = time.time() - countdown_start
        countdown = int(3 - elapsed) + 1

        if elapsed < 3:
            cv2.putText(frame, f"Show Gesture In: {countdown}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            if computer_choice is None:
                computer_choice = random.choice(choices)
                result_game = winner(user_move, computer_choice)

                if result_game == "User":
                    user_score += 1
                    round_result = "You Win!"
                elif result_game == "Computer":
                    computer_score += 1
                    round_result = "Computer Wins!"
                else:
                    round_result = "It's a Tie!"

            cv2.putText(frame, f"Your Move: {user_move}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Computer: {computer_choice}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, round_result, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

            if elapsed > 5:
                round_active = False

    cv2.putText(frame, f"Score  You: {user_score}   Computer: {computer_score}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Rock Paper Scissors - Fair Play Mode (ESC to Exit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
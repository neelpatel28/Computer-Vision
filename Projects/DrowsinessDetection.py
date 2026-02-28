import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]

EYE_CLOSED_THRESH = 0.25
DROWSY_TIME = 2
YAWN_THRESH = 25

sleep_start = None

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 130), (20, 20, 20), -1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status_text = "Status: Awake"
    status_color = (0, 255, 0)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE])
            EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            mouth_open = abs(landmarks[MOUTH[0]].y - landmarks[MOUTH[1]].y) * h

            if EAR < EYE_CLOSED_THRESH:
                if sleep_start is None:
                    sleep_start = time.time()
                elif time.time() - sleep_start > DROWSY_TIME:
                    status_text = "ALERT: Driver Drowsy!"
                    status_color = (0, 0, 255)
            else:
                sleep_start = None

            if mouth_open > YAWN_THRESH:
                status_text = "Warning: Yawning Detected"
                status_color = (0, 165, 255)

    cv2.putText(panel, "Driver Drowsiness Monitoring System", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(panel, status_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    frame = cv2.addWeighted(panel, 0.7, frame, 0.3, 0)
    cv2.imshow("Drowsiness Detection (Press ESC to Exit)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
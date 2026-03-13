import cv2
import mediapipe as mp
import time
import numpy as np

# ==============================
# KONFIGURACJA MODELU
# ==============================

model_path = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

# ==============================
# ZMIENNE DO DETEKCJI MACHANIA
# ==============================

prev_right_wrist_x = None
prev_left_wrist_x = None

right_wave_counter = 0
left_wave_counter = 0
last_wave_time = 0
DISPLAY_DURATION = 1  # sekundy

# ==============================
# START KAMERY
# ==============================

cap = cv2.VideoCapture(1)
start_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]
            h, w, _ = frame.shape

            # ==============================
            # RYSOWANIE SZKIELETU
            # ==============================
            for landmark in landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            for connection in POSE_CONNECTIONS:
                start = landmarks[connection.start]
                end = landmarks[connection.end]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ==============================
            # DETEKCJA MACHANIA
            # ==============================

            right_wrist = landmarks[16]
            left_wrist = landmarks[15]

            right_x = right_wrist.x
            left_x = left_wrist.x

            # -------- PRAWĄ RĘKA --------
            if prev_right_wrist_x is not None:
                movement = abs(right_x - prev_right_wrist_x)

                if movement > 0.04:
                    right_wave_counter += 1
                else:
                    right_wave_counter = 0

                if right_wave_counter > 2:
                    last_wave_time = time.time()

            prev_right_wrist_x = right_x


            # -------- LEWĄ RĘKA --------
            if prev_left_wrist_x is not None:
                movement = abs(left_x - prev_left_wrist_x)

                if movement > 0.04:
                    left_wave_counter += 1
                else:
                    left_wave_counter = 0

                if left_wave_counter > 2:
                    last_wave_time = time.time()

            prev_left_wrist_x = left_x
            
            

        # ==============================
        # WYŚWIETLANIE DYMKA (3 sekundy)
        # ==============================

        if time.time() - last_wave_time < DISPLAY_DURATION:

            bubble_x = 50
            bubble_y = 50
            bubble_w = 350
            bubble_h = 150

            # Prostokąt dymka
            cv2.rectangle(
                frame,
                (bubble_x, bubble_y),
                (bubble_x + bubble_w, bubble_y + bubble_h),
                (255, 255, 255),
                -1
            )

            # Obramowanie
            cv2.rectangle(
                frame,
                (bubble_x, bubble_y),
                (bubble_x + bubble_w, bubble_y + bubble_h),
                (0, 0, 0),
                3
            )

            # Trójkąt "ogonka"
            points = [
                (bubble_x + 60, bubble_y + bubble_h),
                (bubble_x + 90, bubble_y + bubble_h),
                (bubble_x + 75, bubble_y + bubble_h + 30)
            ]
            cv2.fillPoly(frame, [np.array(points)], (255, 255, 255))
            cv2.polylines(frame, [np.array(points)], True, (0, 0, 0), 3)

            # Tekst w środku
            cv2.putText(
                frame,
                "Siemka",
                (bubble_x + 40, bubble_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

        cv2.imshow("Wave to interact :)  (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

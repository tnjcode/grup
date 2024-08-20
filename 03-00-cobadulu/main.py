from collections import deque
from ultralytics import YOLO
import numpy as np
import cv2
import time
import os

# Konfigurasi video dan buffer
buffer_size = 32
output_folder = "C:\\Users\\LENOVO\\grup\\VAR\\dataP"

# Load YOLOv8 Model
yolo = YOLO('yolov8s.pt')

# Inisialisasi variabel
pts = deque(maxlen=buffer_size)
frame_buffer = deque(maxlen=buffer_size)  # Buffer untuk menyimpan frame
score = 0
goal_counted = False
recording = False
out = None
record_count = 1
goal_time = None
start_time = time.time()  # Waktu mulai dari 0
slowmo_factor = 8  # Faktor slowmo, 2 berarti 2x lebih lambat

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    height, width = frame.shape[:2]

    # Garis biru sebagai garis pre-goal dan garis hijau sebagai garis goal
    blue_line_y = height - 150  
    green_line_y = height - 50  

    # Gambar garis biru (pre-goal) dan garis hijau (goal)
    cv2.line(frame, (0, blue_line_y), (width, blue_line_y), (255, 0, 0), 2)
    cv2.line(frame, (0, green_line_y), (width, green_line_y), (0, 255, 0), 2)

    # Menampilkan waktu real-time di window
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    current_time_text = f"Time: {minutes:02}:{seconds:02}"
    cv2.putText(frame, current_time_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # YOLOv8 Deteksi
    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                class_name = classes_names[cls]

                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                if class_name == "sports ball":
                    center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
                    radius = (x2 - x1) // 2

                    # Memulai rekaman jika bola melewati garis biru (pre-goal)
                    if y2 > blue_line_y and not recording:
                        video_name = os.path.join(output_folder, f"rekaman_{record_count}.avi")
                        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
                        print(f"Started recording: {video_name}")
                        recording = True
                        record_count += 1

                    # Setelah bola melewati garis hijau (goal)
                    if y2 > green_line_y and recording and not goal_counted:
                        goal_time = elapsed_time
                        score += 1
                        goal_counted = True
                        print(f"Goal at {goal_time:.2f} sec")

                    if recording:
                        if goal_counted:
                            cv2.putText(frame, f"Goal: {goal_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
                            cv2.putText(frame, f"Menit: {int(goal_time // 60)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
                            cv2.putText(frame, f"Detik: {int(goal_time % 60)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
                        
                        # Tambahkan frame tambahan untuk slow motion
                        for _ in range(slowmo_factor):
                            out.write(frame)

    # Tambahkan frame ke buffer untuk tayangan ulang
    frame_buffer.append(frame)

    cv2.putText(frame, f"Score: {score}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if goal_time is not None:
        cv2.putText(frame, f"Goal Time: {goal_time:.2f} sec", (width - 250, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if goal_counted:
        time.sleep(3)
        out.release()  # Stop and save video
        print("Video saved")
        
        # Tampilkan tayangan ulang dalam slow motion
        for replay_frame in frame_buffer:  # Tampilkan frame dalam urutan maju
            for _ in range(slowmo_factor):  # Ulangi setiap frame untuk efek slowmo
                slow_frame = cv2.resize(replay_frame, None, fx=1, fy=1)
                cv2.imshow('frame', slow_frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):  # Penundaan untuk slow-motion
                    break

        goal_counted = False  # Reset flag setelah jeda
        recording = False  # Set recording to False to start new recording for next goal
        frame_buffer.clear()  # Bersihkan buffer setelah tayangan ulang

videoCap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

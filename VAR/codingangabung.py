from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import os

# Konfigurasi video dan buffer
buffer_size = 32
output_folder = "C:\\Users\\LENOVO\\grup\\VAR\\dataP"

# Load YOLO Model
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
labels = open("coco.names").read().strip().split("\n")

# Inisialisasi variabel
pts = deque(maxlen=buffer_size)
counter = 0
(dX, dY) = (0, 0)
direction = ""
score = 0
goal_counted = False
recording = False
out = None
record_count = 1
goal_time = None
start_time = time.time()  # Waktu mulai dari 0

# Mulai video stream dari webcam
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop utama
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]

    # Proses YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    blue_line_y = height - 150  # Garis biru sebagai garis pre-goal
    green_line_y = height - 50  # Garis hijau sebagai garis goal

    # Gambar garis biru (pre-goal) dan garis hijau (goal)
    cv2.line(frame, (0, blue_line_y), (width, blue_line_y), (255, 0, 0), 2)
    cv2.line(frame, (0, green_line_y), (width, green_line_y), (0, 255, 0), 2)
    
    # Menampilkan waktu real-time di window
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    current_time_text = f"Time: {minutes:02}:{seconds:02}"
    cv2.putText(frame, current_time_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            confidence = confidences[i]

            if label == "sports ball":  # Deteksi objek bola
                center = (x + w // 2, y + h // 2)
                radius = w // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)

                # Memulai rekaman jika bola melewati garis biru (pre-goal)
                if int(y + h) > blue_line_y and not recording:
                    video_name = os.path.join(output_folder, f"rekaman_{record_count}.avi")
                    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
                    print(f"Started recording: {video_name}")
                    recording = True
                    record_count += 1

                # Setelah bola melewati garis hijau (goal)
                if int(y + h) > green_line_y and recording and not goal_counted:
                    goal_time = elapsed_time
                    score += 1
                    goal_counted = True
                    print(f"Goal at {goal_time:.2f} sec")

                # Jika rekaman aktif, simpan frame ke video
                if recording:
                    # Menampilkan informasi goal
                    if goal_counted:
                        cv2.putText(frame, f"Goal: {goal_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)
                        cv2.putText(frame, f"Menit: {int(goal_time // 60)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)
                        cv2.putText(frame, f"Detik: {int(goal_time % 60)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)
                    
                    out.write(frame)

    else:
        if recording:
            recording = False
            out.release()
            print("Stopped recording")

    # Menampilkan skor dan waktu goal terakhir
    cv2.putText(frame, f"Score: {score}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if goal_time is not None:
        cv2.putText(frame, f"Goal Time: {goal_time:.2f} sec", (width - 250, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    # Jika 'q' ditekan, berhenti
    if key == ord("q"):
        break

    # Jeda 3 detik setelah 'goal' sambil terus merekam
    if goal_counted:
        time.sleep(3)
        out.release()  # Stop and save video
        print("Video saved")
        goal_counted = False  # Reset flag setelah jeda

# Memberhentikan stream dan menutup semua jendela
vs.stop()
if out is not None:
    out.release()
cv2.destroyAllWindows()

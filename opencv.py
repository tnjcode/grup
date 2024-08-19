from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import os

# Konfigurasi video dan buffer
buffer_size = 32
output_folder = "C:\\SMA_PRAXIS\\03-00-cobadulu\\RekamanGoal"

# Load YOLO model
yolo_weights = 'yolov3.weights'
yolo_config = 'yolov3.cfg'
yolo_labels = 'coco.names'

net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load label class dari COCO dataset
with open(yolo_labels, 'rt') as f:
    labels = f.read().strip().split("\n")

# Inisialisasi variabel
pts = deque(maxlen=buffer_size)
counter = 0
(dX, dY) = (0, 0)
direction = ""
score = 0
goal_counted = False
slowmo_factor = 3
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
    
    # Preprocessing YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Proses deteksi
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and labels[classID] == "sports ball":  # Pastikan mendeteksi bola
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-Maxima Suppression untuk mengeliminasi box yang tumpang tindih
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    center = None

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            mid_x = width // 5

            center = (x + w // 2, y + h // 2)

            # Gambarkan kotak di sekitar bola
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{labels[classIDs[i]]}: {confidences[i]:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            pts.appendleft(center)

            # Memulai rekaman jika bola melewati garis
            if int(center[0] + w / 2) < mid_x and not goal_counted:
                if not recording:
                    video_name = os.path.join(output_folder, f"rekaman_{record_count}.avi")
                    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
                    print(f"Started recording: {video_name}")
                    recording = True
                    record_count += 1

            if int(center[0] + w / 2) >= mid_x and recording:
                # Mencatat waktu goal
                elapsed_time = time.time() - start_time

                goal_time = elapsed_time
                cv2.putText(frame, f"Goal: {goal_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)
                score += 1
                goal_counted = True
                print(f"Goal at {goal_time:.2f} sec")

            elif int(center[0] + w / 2) >= mid_x:
                goal_counted = False

            # Tuliskan frame ke video dengan efek slow motion
            if recording:
                for _ in range(slowmo_factor):
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

    # Jeda 5 detik setelah 'goal'
    if goal_counted:
        time.sleep(5)
        goal_counted = False  # Reset flag setelah jeda

# Memberhentikan stream dan menutup semua jendela
vs.stop()
if out is not None:
    out.release()
cv2.destroyAllWindows()

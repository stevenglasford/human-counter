
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
from collections import deque
from datetime import datetime

# Configuration
VIDEO_PATH = "your_video.mp4"  # Replace with your 8K video path
OUTPUT_DIR = "cv_output"
CONFIDENCE_THRESHOLD = 0.5
STOP_SPEED_THRESHOLD = 0.5  # pixels/frame considered as stopped
STOP_DURATION_THRESHOLD = 15  # frames

# Load YOLOv11x (make sure you have the model locally or adjust to load from hub)
model = YOLO("yolo11x.pt")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tracking state
vehicle_tracks = {}  # id -> deque of (frame_idx, x, y)
stops_detected = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)

    # Process detections
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes.xyxy))

        for i, box in enumerate(boxes.xyxy):
            cls = int(boxes.cls[i].item())
            if cls in [2, 3, 5, 7]:  # vehicle classes in COCO: car, motorcycle, bus, truck
                x1, y1, x2, y2 = box.cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                vid = ids[i]

                if vid not in vehicle_tracks:
                    vehicle_tracks[vid] = deque(maxlen=STOP_DURATION_THRESHOLD + 5)
                vehicle_tracks[vid].append((frame_idx, cx, cy))

                # Detect stopping
                track = vehicle_tracks[vid]
                if len(track) >= STOP_DURATION_THRESHOLD:
                    speeds = [np.linalg.norm(np.array(track[i][1:]) - np.array(track[i-1][1:])) for i in range(1, len(track))]
                    avg_speed = sum(speeds) / len(speeds)
                    if avg_speed < STOP_SPEED_THRESHOLD:
                        stops_detected.append((vid, frame_idx, cx, cy, avg_speed))

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()

# Output results
df = pd.DataFrame(stops_detected, columns=["VehicleID", "Frame", "X", "Y", "AvgSpeed"])
df.to_csv(os.path.join(OUTPUT_DIR, "stops_detected.csv"), index=False)

print(f"Done. Detected {len(stops_detected)} stop events. Results saved to {OUTPUT_DIR}/stops_detected.csv")

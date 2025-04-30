
import os
import sys
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import xml.etree.ElementTree as ET

# Configuration
YOLO_MODEL_PATH = "yolo11x.pt"
CONFIDENCE_THRESHOLD = 0.5
STOP_SPEED_THRESHOLD = 0.5  # pixels/frame
STOP_DURATION_THRESHOLD = 15  # frames
VEHICLE_CLASSES = [2, 3, 5, 7]
TRAFFIC_LIGHT_CLASS = 9

def parse_gpx(gpx_file):
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    ns = {'default': 'http://www.topografix.com/GPX/1/1'}
    trkpts = root.findall('.//default:trkpt', ns)
    gpx_data = []
    for pt in trkpts:
        lat = float(pt.attrib['lat'])
        lon = float(pt.attrib['lon'])
        time_elem = pt.find('default:time', ns)
        if time_elem is not None:
            timestamp = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
            gpx_data.append((timestamp, lat, lon))
    return gpx_data

def sync_gpx_to_frames(gpx_data, video_start_time, fps):
    synced = {}
    for (timestamp, lat, lon) in gpx_data:
        frame_num = int((timestamp - video_start_time).total_seconds() * fps)
        synced[frame_num] = (lat, lon)
    return synced

def main():
    if len(sys.argv) < 2:
        print("Usage: python yolo11x_full_analysis_with_gpx.py <video_file>")
        sys.exit(1)

    VIDEO_PATH = sys.argv[1]
    video_dir = os.path.dirname(VIDEO_PATH)
    OUTPUT_DIR = os.path.join(video_dir, "cv_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find GPX file in same directory
    gpx_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.gpx')]
    if len(gpx_files) != 1:
        print("Error: Expected exactly one GPX file in the video directory.")
        sys.exit(1)

    GPX_PATH = os.path.join(video_dir, gpx_files[0])

    # Manually set this to match video start time
    VIDEO_START_TIME = datetime(2025, 4, 1, 12, 0, 0)

    # Load GPX data and sync to frames
    gpx_data = parse_gpx(GPX_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    gpx_by_frame = sync_gpx_to_frames(gpx_data, VIDEO_START_TIME, fps)

    model = YOLO(YOLO_MODEL_PATH)

    vehicle_tracks = {}
    stops_detected = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
        lights_in_frame = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes.xyxy))

            for i, box in enumerate(boxes.xyxy):
                cls = int(boxes.cls[i].item())
                x1, y1, x2, y2 = box.cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if cls == TRAFFIC_LIGHT_CLASS:
                    lights_in_frame.append((cx, cy))
                elif cls in VEHICLE_CLASSES:
                    vid = ids[i]
                    if vid not in vehicle_tracks:
                        vehicle_tracks[vid] = deque(maxlen=STOP_DURATION_THRESHOLD + 5)
                    vehicle_tracks[vid].append((frame_idx, cx, cy))
                    track = vehicle_tracks[vid]
                    if len(track) >= STOP_DURATION_THRESHOLD:
                        speeds = [np.linalg.norm(np.array(track[i][1:]) - np.array(track[i-1][1:])) for i in range(1, len(track))]
                        avg_speed = sum(speeds) / len(speeds)
                        if avg_speed < STOP_SPEED_THRESHOLD:
                            lat, lon = gpx_by_frame.get(frame_idx, (None, None))
                            stops_detected.append((vid, frame_idx, cx, cy, avg_speed, lat, lon, len(lights_in_frame)))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()

    # Output
    df = pd.DataFrame(stops_detected, columns=[
        "VehicleID", "Frame", "X", "Y", "AvgSpeed", "Latitude", "Longitude", "TrafficLightsVisible"
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "stops_detected.csv"), index=False)

    total_stops = len(df)
    avg_speed_during_stops = df["AvgSpeed"].mean() if not df.empty else 0
    avg_lights_per_stop = df["TrafficLightsVisible"].mean() if not df.empty else 0

    report = f"""
TRAFFIC STOP ANALYSIS REPORT
Video: {VIDEO_PATH}
GPX: {GPX_PATH}
FPS: {fps}
Total Frames Processed: {frame_idx}
Video Start Time: {VIDEO_START_TIME}

--- DETECTION RESULTS ---
Total Stops Detected: {total_stops}
Average Speed During Stops: {avg_speed_during_stops:.2f} pixels/frame
Average Visible Traffic Lights at Stop: {avg_lights_per_stop:.2f}

Interpretation:
- Stops near traffic lights with low average speed likely indicate red light events.
- Use the GPX coordinates to map high-frequency stop zones for city planning.
"""

    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
        f.write(report)

    print("Complete. Outputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

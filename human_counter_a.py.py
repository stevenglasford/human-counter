
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from norfair import Detection, Tracker
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from multiprocessing import Pool
import os
import sys
import glob
import math

MOVEMENT_THRESHOLD = 20
PIXELS_PER_FOOT = 5.0

def parse_gpx(gpx_path):
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = {'default': 'http://www.topografix.com/GPX/1/1'}
    trkpts = root.findall('.//default:trkpt', ns)
    times, coords = [], []
    for pt in trkpts:
        lat = float(pt.attrib['lat'])
        lon = float(pt.attrib['lon'])
        time = pt.find('default:time', ns)
        if time is not None:
            times.append(datetime.fromisoformat(time.text.replace('Z', '+00:00')))
            coords.append((lat, lon))
    return times, coords

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def compute_gpx_speeds(times, coords):
    speeds = []
    for i in range(1, len(times)):
        dt = (times[i] - times[i-1]).total_seconds()
        dist = haversine(*coords[i-1], *coords[i])
        speed = dist / dt if dt > 0 else 0
        speeds.append((times[i], speed * 2.23694))  # convert m/s to mph
    return speeds

def get_avg_speed_and_series(gpx_file):
    times, coords = parse_gpx(gpx_file)
    if len(times) < 2:
        return 0, pd.Series()
    speeds = compute_gpx_speeds(times, coords)
    avg_speed = np.mean([s[1] for s in speeds])
    speed_series = pd.Series({int((t - times[0]).total_seconds()): s for t, s in speeds})
    return avg_speed, speed_series

def norfair_detections_from_yolo(results, class_id):
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == class_id:
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            detections.append(Detection(points=np.array([[cx, cy]]), scores=np.array([conf.item()])))
    return detections

def euclidean_distance(a, b):
    a_coords = a.estimate if hasattr(a, 'estimate') else a.points
    b_coords = b.estimate if hasattr(b, 'estimate') else b.points
    return np.linalg.norm(a_coords - b_coords)

def process_video(video_path):
    folder = os.path.dirname(video_path)
    gpx_files = [f for f in os.listdir(folder) if f.endswith(".gpx")]
    gpx_file = os.path.join(folder, gpx_files[0]) if gpx_files else None
    gpx_avg_speed, _ = get_avg_speed_and_series(gpx_file) if gpx_file else (0, pd.Series())

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    model = YOLO("yolo11x.pt")
    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

    idle_counts, walking_counts, total_counts, car_speeds, confidences = [], [], [], [], []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        detections = norfair_detections_from_yolo(results, 0)
        tracked = tracker.update(detections)

        conf = results.boxes.conf
        if len(conf) > 0:
            confidences.append(conf.mean().item())

        idle = walking = 0
        for t in tracked:
            speed = np.linalg.norm(t.estimate - t.previous_estimate) if t.previous_estimate is not None else 0
            (idle if speed < MOVEMENT_THRESHOLD else walking) += 1

        idle_counts.append(idle)
        walking_counts.append(walking)
        total_counts.append(idle + walking)
        frame_count += 1

    cap.release()

    seconds = list(range(len(total_counts)))
    avg_conf = np.mean(confidences) * 100
    df = pd.DataFrame({
        "second": seconds,
        "idle_count": idle_counts,
        "walking_count": walking_counts,
        "total_count": total_counts
    })

    plt.figure(figsize=(14, 6))
    plt.plot(df["second"], df["idle_count"], label="Idle", color="brown")
    plt.plot(df["second"], df["walking_count"], label="Walking", color="blue")
    plt.plot(df["second"], df["total_count"], label="Total", color="black")
    plt.axhline(y=avg_conf, color='red', linestyle='--', label="Avg Detection Accuracy")
    plt.axhline(y=gpx_avg_speed, color='brown', linestyle='--', label="Avg GPX Speed (mph)")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Count / % / MPH")
    plt.title("Human Detection with GPX Adjustment and Accuracy Overlay")
    plt.legend()
    plt.grid(True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    plt.savefig(f"{base}__graph.png")
    df.to_csv(f"{base}__data.csv", index=False)
    return f"Processed {video_path}"

def run_batch(video_paths):
    with Pool() as pool:
        results = pool.map(process_video, video_paths)
    for r in results:
        print(r)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python human_counter.py video1.mp4 video2.mp4")
    else:
        video_files = [f for f in sys.argv[1:] if f.endswith(".mp4") or f.endswith(".MP4")]
        run_batch(video_files)

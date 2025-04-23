
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
from datetime import timezone
video_start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)

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
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def compute_gpx_speed_series(times, coords):
    speed_dict = {}
    for i in range(1, len(times)):
        dt = (times[i] - times[i-1]).total_seconds()
        dist = haversine(*coords[i-1], *coords[i])
        speed = dist / dt if dt > 0 else 0
        timestamp = times[i]
        speed_dict[timestamp] = speed * 2.23694  # m/s to mph
    return pd.Series(speed_dict)

def match_gpx_to_video(gpx_series, video_start_time, video_duration):
    gpx_times = gpx_series.index
    aligned_speeds = []
    for i in range(video_duration):
        t = video_start_time + timedelta(seconds=i)
        if t in gpx_series:
            aligned_speeds.append(gpx_series[t])
        else:
            # Interpolate from nearest values
            before = gpx_series[gpx_series.index <= t]
            after = gpx_series[gpx_series.index >= t]
            if not before.empty and not after.empty:
                t0, v0 = before.index[-1], before.iloc[-1]
                t1, v1 = after.index[0], after.iloc[0]
                weight = (t - t0).total_seconds() / (t1 - t0).total_seconds() if (t1 - t0).total_seconds() != 0 else 0
                interpolated = v0 + (v1 - v0) * weight
                aligned_speeds.append(interpolated)
            else:
                aligned_speeds.append(np.nan)
    return aligned_speeds

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

def get_video_start_time(video_path):
    base = os.path.basename(video_path)
    try:
        timestamp_str = base.split("__")[1].split(".")[0]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except:
        return datetime.now()

def process_video(video_path):
    folder = os.path.dirname(video_path)
    gpx_files = [f for f in os.listdir(folder) if f.endswith(".gpx")]
    gpx_file = os.path.join(folder, gpx_files[0]) if gpx_files else None

    video_start_time = get_video_start_time(video_path)
    
    gpx_series = pd.Series()
    if gpx_file:
        times, coords = parse_gpx(gpx_file)
        gpx_series = compute_gpx_speed_series(times, coords)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps
    model = YOLO("yolo11x.pt")
    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

    gpx_speeds = match_gpx_to_video(gpx_series, video_start_time, duration) if not gpx_series.empty else [0]*duration
    avg_gpx_speed = np.nanmean(gpx_speeds) if gpx_speeds else 0

    idle_counts, walking_counts, total_counts, confidences = [], [], [], []
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
        current_sec = frame_count // fps
        self_speed = gpx_speeds[current_sec] if current_sec < len(gpx_speeds) else 0

        for t in tracked:
            speed = np.linalg.norm(t.estimate - t.previous_estimate) if t.previous_estimate is not None else 0
            adjusted_speed = abs(speed - self_speed)
            if adjusted_speed < MOVEMENT_THRESHOLD:
                idle += 1
            else:
                walking += 1

        idle_counts.append(idle)
        walking_counts.append(walking)
        total_counts.append(idle + walking)
        frame_count += 1

    cap.release()
    seconds = list(range(len(total_counts)))
    avg_conf = np.mean(confidences) * 100 if confidences else 0

    df = pd.DataFrame({
        "second": seconds,
        "idle_count": idle_counts,
        "walking_count": walking_counts,
        "total_count": total_counts
    })

    base = os.path.splitext(os.path.basename(video_path))[0]
    plt.figure(figsize=(14, 6))
    plt.plot(df["second"], df["idle_count"], label="Idle", color="brown")
    plt.plot(df["second"], df["walking_count"], label="Walking", color="blue")
    plt.plot(df["second"], df["total_count"], label="Total", color="black")
    plt.axhline(y=avg_conf, color='red', linestyle='--', label="Avg Detection Accuracy")
    plt.axhline(y=avg_gpx_speed, color='brown', linestyle='--', label="Avg GPX Speed (mph)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count / % / MPH")
    plt.title("Human Detection with Timestamp-Synced GPX and Accuracy Overlay")
    plt.legend()
    plt.grid(True)
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

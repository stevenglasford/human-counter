
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from norfair import Detection, Tracker
from multiprocessing import Pool
import os
import datetime
import sys
import glob

model = YOLO("yolo11x.pt")
MOVEMENT_THRESHOLD = 20
PIXELS_PER_FOOT = 5.0

def detect_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 80) & (v > 100)
    hue = h[mask]
    if hue.size == 0:
        return 'unknown'
    hist = cv2.calcHist([hue.astype(np.uint8)], [0], None, [180], [0, 180])
    dominant = int(hist.argmax())
    if 0 <= dominant <= 10 or 160 <= dominant <= 180:
        return 'red'
    elif 15 <= dominant <= 35:
        return 'yellow'
    elif 36 <= dominant <= 89:
        return 'green'
    else:
        return 'unknown'

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Failed to open video: {video_path}"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    idle_counts, walking_counts, total_counts, car_speeds = [], [], [], []
    light_states, red_violations, bus_waiting = [], [], []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start_millis = cap.get(cv2.CAP_PROP_POS_MSEC)
    start_time = datetime.datetime.fromtimestamp(start_millis / 1000.0) if start_millis > 0 else datetime.datetime.now()
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

    human_tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)
    car_tracker = Tracker(distance_function=euclidean_distance, distance_threshold=50)
    red_active = red_violated = bus_seen = False
    current_light = 'unknown'
    person_histories, car_histories = {}, {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        human_dets = norfair_detections_from_yolo(results, class_id=0)
        car_dets = norfair_detections_from_yolo(results, class_id=2)
        tracked_humans = human_tracker.update(human_dets)
        tracked_cars = car_tracker.update(car_dets)

        for t in tracked_humans:
            pid = t.id
            cx, cy = t.estimate[0]
            person_histories.setdefault(pid, []).append((frame_count, cx, cy))

        for t in tracked_cars:
            cid = t.id
            cx, cy = t.estimate[0]
            car_histories.setdefault(cid, []).append((frame_count, cx, cy))

        red_now = yellow_now = green_now = False
        bus_now = any(int(c) == 5 for c in results.boxes.cls)
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 9:
                x1, y1, x2, y2 = map(int, box)
                if y2 > frame.shape[0] * 0.6 or (x2 - x1)/(y2 - y1) > 1.5:
                    continue
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    state = detect_light_color(roi)
                    if state == 'red': red_now = True
                    elif state == 'yellow': yellow_now = True
                    elif state == 'green': green_now = True

        if red_now:
            current_light = 'red'
            red_active = True
            if len(tracked_humans) >= 5:
                red_violated = True
            if bus_now:
                bus_seen = True
        elif yellow_now: current_light = 'yellow'
        elif green_now: current_light = 'green'
        else: current_light = 'unknown'

        frame_count += 1
        if frame_count % fps == 0:
            idle = walking = 0
            mph_list = []
            recent_frame = frame_count - 1

            for pid, history in person_histories.items():
                recent = [(x, y) for f, x, y in history if recent_frame - fps <= f <= recent_frame]
                if len(recent) >= 2:
                    d = sum(np.linalg.norm(np.array(recent[i]) - np.array(recent[i - 1])) for i in range(1, len(recent)))
                    avg_speed = d / len(recent)
                    if avg_speed < MOVEMENT_THRESHOLD:
                        idle += 1
                    else:
                        walking += 1

            for cid, history in car_histories.items():
                recent = [(x, y) for f, x, y in history if recent_frame - fps <= f <= recent_frame]
                if len(recent) >= 2:
                    d = sum(np.linalg.norm(np.array(recent[i]) - np.array(recent[i - 1])) for i in range(1, len(recent)))
                    avg_pixels_per_sec = d / len(recent)
                    feet_per_sec = avg_pixels_per_sec / PIXELS_PER_FOOT
                    mph = feet_per_sec * 0.681818
                    mph_list.append(mph)

            avg_mph = np.mean(mph_list) if mph_list else 0

            idle_counts.append(idle)
            walking_counts.append(walking)
            total_counts.append(idle + walking)
            car_speeds.append(avg_mph)
            light_states.append(current_light)
            red_violations.append(red_violated)
            bus_waiting.append(bus_seen if red_active else False)
            red_active = red_violated = bus_seen = False
            person_histories = {}
            car_histories = {}

    cap.release()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = f"{video_name}__{timestamp_str}"
    csv_path = f"{output_base}.csv"
    png_path = f"{output_base}.png"

    df = pd.DataFrame({
        "second": list(range(len(total_counts))),
        "idle_count": idle_counts,
        "walking_count": walking_counts,
        "total_count": total_counts,
        "car_speed_mph": car_speeds,
        "light": light_states,
        "red_violation": red_violations,
        "bus_waiting": bus_waiting
    })
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(14, 6))
    for i, row in df.iterrows():
        sec = row["second"]
        light = row["light"]
        violation = row["red_violation"]
        bus = row["bus_waiting"]
        if light == "red":
            if violation and bus:
                plt.axvline(x=sec, color='k', linestyle='--', linewidth=2)
                plt.axvline(x=sec, color='r', linestyle='--', linewidth=1.5)
            elif violation:
                plt.axvline(x=sec, color='r', linestyle='--', linewidth=1.5)
            elif bus:
                plt.axvline(x=sec, color='k', linestyle='-', linewidth=2)
                plt.axvline(x=sec, color='r', linestyle='-', linewidth=1.5)
            else:
                plt.axvline(x=sec, color='r', linestyle='-', linewidth=1.5)
        elif light == "yellow":
            plt.axvline(x=sec, color='y', linestyle='-', linewidth=1.5)
        elif light == "green":
            plt.axvline(x=sec, color='g', linestyle='-', linewidth=1.5)

    plt.plot(df["second"].to_numpy(), df["idle_count"].to_numpy(), label="Idle", color="brown", linewidth=2)
    plt.plot(df["second"].to_numpy(), df["walking_count"].to_numpy(), label="Walking", color="blue", linewidth=2)
    plt.plot(df["second"].to_numpy(), df["total_count"].to_numpy(), label="Total", color="black", linewidth=2)
    plt.plot(df["second"].to_numpy(), df["car_speed_mph"].to_numpy(), label="Car Speed (mph)", color="orange", linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Counts / Speed")
    plt.title(f"Human & Vehicle Tracking - {video_name}")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(png_path)
    plt.close()

    return f"Processed {video_path} → {csv_path}, {png_path}"

def run_batch(video_paths):
    with Pool() as pool:
        results = pool.map(process_video, video_paths)
    for res in results:
        print(res)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python human_counter.py /path/to/videos/*.mp4")
    else:
        video_files = []
        for pattern in sys.argv[1:]:
            video_files.extend(glob.glob(pattern))
        run_batch(video_files)

import cv2
import gpxpy
import numpy as np
import csv
from datetime import timedelta
from scipy.signal import correlate
import argparse

def extract_motion_from_video(video_path, sample_rate=10):
    cap = cv2.VideoCapture(video_path)
    motion_signal = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_signal.append(np.mean(magnitude))
            prev_gray = gray

        frame_idx += 1

    cap.release()
    time_series = np.linspace(0, frame_idx / cap.get(cv2.CAP_PROP_FPS), len(motion_signal))
    return time_series, np.array(motion_signal)

def extract_speed_from_gpx(gpx_file_path):
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points, times = [], []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append((p.latitude, p.longitude))
                times.append(p.time)

    speeds, timestamps = [], []
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
        time_diff = (times[i] - times[i-1]).total_seconds()
        if time_diff > 0:
            speed = dist / time_diff
            speeds.append(speed)
            timestamps.append(times[i])

    time_series = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    return time_series, np.array(speeds), timestamps[0], list(zip(timestamps, points))

def estimate_offset(video_motion, gpx_speed):
    video_signal = (video_motion[1] - np.mean(video_motion[1])) / np.std(video_motion[1])
    gpx_signal = (gpx_speed[1] - np.mean(gpx_speed[1])) / np.std(gpx_speed[1])
    correlation = correlate(gpx_signal, video_signal, mode='full')
    lag = np.argmax(correlation) - len(video_signal) + 1
    offset_seconds = gpx_speed[0][0] + lag
    return offset_seconds

def find_nearest_gpx_point(gpx_data, video_timestamp):
    return min(gpx_data, key=lambda x: abs((x[0] - video_timestamp).total_seconds()))

def overlay_timestamps_on_video(video_path, gpx_data, offset_seconds, gpx_start, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    video_start_time = gpx_start - timedelta(seconds=offset_seconds)

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'video_time', 'gpx_time', 'latitude', 'longitude'])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_time = video_start_time + timedelta(seconds=frame_idx / fps)
            gpx_point = find_nearest_gpx_point(gpx_data, video_time)

            cv2.putText(frame, video_time.strftime('%Y-%m-%d %H:%M:%S'), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Lat: {gpx_point[1][0]:.6f}, Lon: {gpx_point[1][1]:.6f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            out.write(frame)
            writer.writerow([
                frame_idx,
                round(frame_idx / fps, 2),
                gpx_point[0].isoformat(),
                gpx_point[1][0],
                gpx_point[1][1]
            ])
            frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_video_path}")
    print(f"✅ Frame-to-GPX CSV saved to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Align a video with a GPX file and add timestamp/GPS overlays.")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("gpx", help="Input GPX file path")
    parser.add_argument("output_video", help="Output video file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    args = parser.parse_args()

    print("📹 Extracting motion from video...")
    video_time, video_motion = extract_motion_from_video(args.video)

    print("📍 Extracting speed from GPX...")
    gpx_time, gpx_speed, gpx_start, full_gpx_data = extract_speed_from_gpx(args.gpx)

    print("🔄 Estimating offset...")
    offset_seconds = estimate_offset((video_time, video_motion), (gpx_time, gpx_speed))
    print(f"🕒 Estimated offset: {offset_seconds:.2f} seconds")

    print("🎞️ Rendering output...")
    overlay_timestamps_on_video(args.video, full_gpx_data, offset_seconds, gpx_start, args.output_video, args.output_csv)

if __name__ == "__main__":
    main()

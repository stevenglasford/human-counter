import numpy as np
import gpxpy
import matplotlib.pyplot as plt
from scipy.signal import correlate
import argparse
import sys

def extract_motion_from_video(video_path):
    # Placeholder for real motion extraction — this simulates it
    duration = 300
    frame_rate = 30
    total_frames = duration * frame_rate
    time_series = np.linspace(0, duration, total_frames)
    motion_signal = np.sin(0.02 * np.pi * time_series) + np.random.normal(0, 0.1, total_frames)
    return time_series, motion_signal

def extract_speed_from_gpx(gpx_file_path):
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    times = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append((p.latitude, p.longitude))
                times.append(p.time)

    speeds = []
    timestamps = []
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
        time_diff = (times[i] - times[i-1]).total_seconds()
        if time_diff > 0:
            speed = dist / time_diff
            speeds.append(speed)
            timestamps.append(times[i])

    time_series = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    return time_series, np.array(speeds)

def match_video_to_gpx(video_motion, gpx_speed):
    video_signal = (video_motion[1] - np.mean(video_motion[1])) / np.std(video_motion[1])
    gpx_signal = (gpx_speed[1] - np.mean(gpx_speed[1])) / np.std(gpx_speed[1])

    correlation = correlate(gpx_signal, video_signal, mode='full')
    lag = np.argmax(correlation) - len(video_signal) + 1

    offset_seconds = gpx_speed[0][0] + lag
    return offset_seconds, correlation

def main():
    parser = argparse.ArgumentParser(description="Synchronize a 360 video with a GPX file using motion analysis.")
    parser.add_argument("video_path", help="Path to the 360 video file (used for motion pattern)")
    parser.add_argument("gpx_file", help="Path to the GPX file with GPS data")

    args = parser.parse_args()

    try:
        print(f"Processing video: {args.video_path}")
        print(f"Processing GPX file: {args.gpx_file}")

        video_time, video_motion = extract_motion_from_video(args.video_path)
        gpx_time, gpx_speed = extract_speed_from_gpx(args.gpx_file)
        offset_seconds, correlation = match_video_to_gpx((video_time, video_motion), (gpx_time, gpx_speed))

        print(f"\n✅ Estimated video start offset in GPX timeline: {offset_seconds:.2f} seconds")

        plt.title("Cross-correlation between video motion and GPX speed")
        plt.plot(correlation)
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

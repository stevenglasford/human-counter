import cv2
import gpxpy
import csv
from datetime import timedelta

def parse_gpx_with_timestamps(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    return [(p.time, p.latitude, p.longitude) 
            for tr in gpx.tracks 
            for seg in tr.segments 
            for p in seg.points]

def find_nearest_gpx_point(gpx_data, video_timestamp):
    return min(gpx_data, key=lambda x: abs((x[0] - video_timestamp).total_seconds()))

def overlay_timestamps_on_video(video_path, gpx_file_path, offset_seconds, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    gpx_data = parse_gpx_with_timestamps(gpx_file_path)
    video_start_time = gpx_data[0][0] - timedelta(seconds=offset_seconds)

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
            cv2.putText(frame, video_time.strftime('%Y-%m-%d %H:%M:%S'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Lat: {gpx_point[1]:.6f}, Lon: {gpx_point[2]:.6f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            out.write(frame)
            writer.writerow([frame_idx, round(frame_idx/fps, 2), gpx_point[0].isoformat(), gpx_point[1], gpx_point[2]])
            frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_video_path}")
    print(f"✅ Frame-to-GPX CSV saved to: {output_csv_path}")

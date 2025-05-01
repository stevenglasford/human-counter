#!/bin/bash

# This script processes .mp4 videos in batches of 5 concurrent threads.
# Each video is processed by main_gpu.py, and logs are stored per video.

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/videos_directory"
    exit 1
fi

VIDEO_DIR="$1"
PYTHON_SCRIPT="human_counter.py"
MAX_JOBS=1


# Create a log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Function to limit number of parallel jobs
function wait_for_available_slot {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# Find all .mp4 files and process them in controlled parallel batches
find "$VIDEO_DIR" -type f \( -iname "*.mp4" -o -iname "*.MP4" \) | while read -r video; do
    wait_for_available_slot

    logfile="$LOG_DIR/$(basename "$video").log"
    echo "Launching processing of $video..."
    python3 "$PYTHON_SCRIPT" "$video" > "$logfile" 2>&1 &

    sleep 1  # optional delay to reduce burst load
done

# Wait for remaining background jobs to finish
wait
echo "All videos processed. Check logs in $LOG_DIR/"


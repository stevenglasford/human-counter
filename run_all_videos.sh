#!/bin/bash

# This script spawns background jobs for each .mp4 file found recursively
# and runs main_gpu.py on each without needing a graphical terminal.

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/videos_directory"
    exit 1
fi

VIDEO_DIR="$1"
PYTHON_SCRIPT="main_gpu.py"

# Create a log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Find all .mp4 files recursively
find "$VIDEO_DIR" -type f \( -iname "*.mp4" -o -iname "*.MP4" \) | while read -r video; do
    logfile="$LOG_DIR/$(basename "$video").log"
    echo "Launching processing of $video..."
    python3 "$PYTHON_SCRIPT" "$video" > "$logfile" 2>&1 &
    sleep 1  # slight delay to not overload the system
done

wait  # wait for all background jobs to complete
echo "All videos launched. Check the logs in $LOG_DIR/"
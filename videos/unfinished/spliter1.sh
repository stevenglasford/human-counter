#!/bin/bash

INPUT="temp_video_252688372923895808.MP4"
BASENAME="temp_video_252688372923895808"
TIMESTAMPS=(
    "00:01:38:16"
    "00:10:12:28"
    "00:16:47:13"
    "00:22:14:27"
    "00:30:42:28"
    "00:38:41:27"
    "00:45:46:30"
    "00:53:11:17"
    "01:00:36:15"
    "01:07:26:13"
)

# Function to convert SMPTE timecode to seconds
smpte_to_seconds() {
    IFS=':' read -r hh mm ss ff <<< "$1"
    # Adjust frame rate to match your video (e.g. 30 fps)
    fps=30
    echo "$hh*3600 + $mm*60 + $ss + $ff/$fps" | bc -l
}

# Create each segment
for ((i=0; i<${#TIMESTAMPS[@]}; i++)); do
    START=$(smpte_to_seconds "${TIMESTAMPS[$i]}")
    
    if (( i+1 < ${#TIMESTAMPS[@]} )); then
        END=$(smpte_to_seconds "${TIMESTAMPS[$((i+1))]}")
        DURATION=$(echo "$END - $START" | bc -l)
    else
        DURATION=""
    fi

    OUTFILE="${BASENAME}_$((i+1)).MP4"

    echo "Creating $OUTFILE from $START seconds"

    if [ -z "$DURATION" ]; then
        ffmpeg -y -ss "$START" -i "$INPUT" -c copy "$OUTFILE"
    else
        ffmpeg -y -ss "$START" -i "$INPUT" -t "$DURATION" -c copy "$OUTFILE"
    fi
done

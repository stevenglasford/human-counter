for f in *.qt; do
    ffmpeg -i "$f" "${f%.qt}.mp4"
done

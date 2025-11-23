for analysis in songs/individual/analysis/*_analysis.json; do
    song_name=$(basename "$analysis" _analysis.json)
    python reelmaker.py create images/ "songs/individual/$song_name.mp3" --analysis "$analysis" --resolution 1080x1920 --fps 60 -o "output/$song_name.mp4"
done

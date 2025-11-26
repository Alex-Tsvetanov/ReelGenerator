#for analysis in songs/individual/analysis/*_analysis.json; do
#    song_name=$(basename "$analysis" _analysis.json)
#    python reelmaker.py create images/ "songs/individual/$song_name.mp3" --analysis "$analysis" --resolution 1080x1920 --fps 60 -o "output/$song_name.mp4"
#done

analysis=songs/individual/analysis/Slxughter\ -\ Fragment\ \(\ Best\ Part\ \ Slowed\ \ Reverb\ \)_analysis.json
song_name=$(basename "$analysis" _analysis.json)
for method in fullspectrum percussive lowfreq demucs ; do
    echo "Generating video for $song_name with method $method";
    python reelmaker.py create images/ "songs/individual/$song_name.mp3" --analysis "$analysis" --resolution 1080x1920 --fps 60 --beat-method $method -o "output/${song_name}_$method.mp4"
done
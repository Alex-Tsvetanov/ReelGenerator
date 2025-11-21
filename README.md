# ReelMaker 🎬🎵

Create viral-style videos by synchronizing images with audio beats and energy peaks. Perfect for TikTok, Instagram Reels, and YouTube Shorts!

## Features

✨ **Intelligent Audio Analysis**
- Automatic tempo (BPM) detection
- Beat tracking
- Energy peak detection for effects and transitions
- Works with any music genre (from slow ballads to fast-paced electronic)

🎨 **Dynamic Visual Effects**
- **Small peaks** trigger image effects:
  - Pulsation/zoom
  - Hue shifting
  - Rotation
  - Blur effects
- **Big peaks** trigger transitions:
  - Fade in/out
  - Slide transitions (left, right, up, down)
  - Zoom in/out transitions

🎥 **Video Generation**
- Portrait (1080x1920) or landscape formats
- Customizable resolution and FPS
- Automatic audio synchronization
- Professional-quality output

## Installation

### Requirements
- Python 3.8+
- ffmpeg (for audio/video processing)

### Install Dependencies

```bash
pip install librosa matplotlib scipy numpy opencv-python Pillow
```

### Install ffmpeg

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Or use chocolatey:
choco install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Quick Start

### 1. Split Audio Collection (Optional)

If you have a large MP3 with multiple songs and a tracklist:

```bash
python split_songs.py
```

This reads `songs/list.txt` and splits `songs/ViralSongs.mp3` into individual songs in `songs/individual/`.

### 2. Analyze Audio

Analyze a single song:

```bash
python reelmaker.py analyze "songs/individual/song.mp3" --visualize
```

This creates:
- `song_analysis.json` - Beat times, peaks, tempo data
- `song_analysis.png` - Visualization of waveform, beats, and peaks

Batch analyze all songs:

```bash
python reelmaker.py batch-analyze "songs/individual/"
```

### 3. Prepare Images

Create a folder with your images:

```
images/
  ├── photo1.jpg
  ├── photo2.png
  ├── photo3.jpg
  └── ...
```

**Tip:** The number of images should be close to the number of big peaks (transitions). Check the analysis output to see how many transitions were detected.

### 4. Create Video

Generate your viral video:

```bash
python reelmaker.py create images/ "songs/individual/song.mp3" -o my_reel.mp4
```

**Portrait (TikTok/Reels):**
```bash
python reelmaker.py create images/ song.mp3 --resolution 1080x1920 -o portrait.mp4
```

**Landscape (YouTube):**
```bash
python reelmaker.py create images/ song.mp3 --resolution 1920x1080 -o landscape.mp4
```

**Square (Instagram):**
```bash
python reelmaker.py create images/ song.mp3 --resolution 1080x1080 -o square.mp4
```

## Usage Examples

### Example 1: Simple Workflow

```bash
# 1. Analyze the audio
python reelmaker.py analyze "songs/viral_song.mp3" --visualize

# 2. Check the analysis output
#    It might say: "Big peaks (transitions): 15"
#    So you'll need ~15-20 images

# 3. Put your images in a folder
mkdir my_images
# Add your images...

# 4. Create the video
python reelmaker.py create my_images/ "songs/viral_song.mp3" -o viral_reel.mp4
```

### Example 2: Batch Process Multiple Songs

```bash
# Analyze all songs in a directory
python reelmaker.py batch-analyze "songs/individual/"

# Create videos for each (you'll need to run this for each song)
python reelmaker.py create images1/ "songs/individual/song1.mp3" -o reel1.mp4
python reelmaker.py create images2/ "songs/individual/song2.mp3" -o reel2.mp4
```

### Example 3: Custom Settings

```bash
# High FPS for smooth effects
python reelmaker.py create images/ song.mp3 --fps 60 -o smooth.mp4

# Custom resolution
python reelmaker.py create images/ song.mp3 --resolution 720x1280 -o custom.mp4

# Specify analysis file manually
python reelmaker.py create images/ song.mp3 --analysis custom_analysis.json -o output.mp4
```

## How It Works

### 1. Audio Analysis (`audio_analyzer.py`)

The system uses `librosa`, a powerful audio analysis library, to:

1. **Load audio** and extract the waveform
2. **Detect tempo** using beat tracking algorithms
3. **Find beats** throughout the song
4. **Calculate onset strength** - measures energy changes
5. **Identify peaks**:
   - **Small peaks** (70th percentile) - frequent, for effects
   - **Big peaks** (90th percentile) - less frequent, for transitions

The algorithm adapts to different tempos automatically:
- Fast songs (120+ BPM) → More frequent effects
- Slow songs (60-90 BPM) → Slower, more dramatic transitions

### 2. Video Generation (`video_generator.py`)

1. **Load images** and resize to target resolution
2. **Plan effects and transitions** based on peak times
3. **Render each frame**:
   - Apply current image
   - Check for active transitions
   - Apply any effects at the current time
4. **Combine with audio** using ffmpeg

### 3. Effects System

**Small Peak Effects** (triggered on beat accents):
- `PULSE` - Scale image in/out
- `HUE_SHIFT` - Adjust color saturation
- `ROTATE` - Slight rotation
- `ZOOM` - Zoom in/out
- `BLUR_PULSE` - Blur effect

**Big Peak Transitions** (triggered on major beats):
- `FADE` - Crossfade between images
- `SLIDE_LEFT/RIGHT/UP/DOWN` - Sliding transitions
- `ZOOM_IN/OUT` - Zoom-based transitions

Effects are automatically distributed across peaks for variety.

## File Structure

```
ReelMaker/
├── reelmaker.py           # Main CLI interface
├── audio_analyzer.py      # Audio analysis engine
├── video_generator.py     # Video rendering engine
├── split_songs.py         # Split large audio files
├── songs/
│   ├── list.txt           # Tracklist for splitting
│   ├── ViralSongs.mp3     # Large audio file (optional)
│   └── individual/        # Individual song files
│       ├── song1.mp3
│       ├── song1_analysis.json
│       ├── song1_analysis.png
│       └── ...
└── images/                # Your images folder
    ├── photo1.jpg
    └── ...
```

## Advanced Usage

### Using as Python Library

```python
from audio_analyzer import AudioAnalyzer
from video_generator import VideoGenerator

# Analyze audio
analyzer = AudioAnalyzer("song.mp3")
results = analyzer.analyze_full()

# Create video
generator = VideoGenerator(
    images_dir="images/",
    audio_file="song.mp3",
    analysis_file="song_analysis.json",
    output_file="output.mp4",
    resolution=(1080, 1920),
    fps=30
)
generator.create_video()
```

### Customizing Effects

Edit `video_generator.py` to:
- Modify effect intensities
- Add new effect types
- Change transition durations
- Adjust peak detection thresholds

### Understanding Analysis Data

The `*_analysis.json` files contain:

```json
{
  "tempo_bpm": 120.5,
  "duration": 180.0,
  "beat_times": [0.5, 1.0, 1.5, ...],
  "small_peak_times": [2.3, 5.7, ...],
  "big_peak_times": [10.2, 25.4, ...],
  "num_beats": 360,
  "num_small_peaks": 45,
  "num_big_peaks": 18
}
```

## Tips for Best Results

1. **Image Quality**: Use high-resolution images (at least 1080p)
2. **Image Count**: Have ~1.5x the number of big peaks for variety
3. **Image Content**: Mix close-ups and wide shots for visual interest
4. **Audio Quality**: Use high-quality audio files (320kbps MP3 or better)
5. **Tempo Matching**: Fast songs work great with many images; slow songs need fewer, more impactful images

## Troubleshooting

**Error: "No images found"**
- Make sure your images are in JPG, PNG, or BMP format
- Check the images directory path

**Error: "Analysis file not found"**
- Run audio analysis first: `python reelmaker.py analyze song.mp3`

**Video has no audio**
- Ensure ffmpeg is installed and in your PATH
- Check audio file format (MP3, WAV, FLAC supported)

**Effects seem off-beat**
- Try re-analyzing with different settings
- Some songs have irregular beats that are harder to detect
- Consider manually editing the analysis JSON file

**Video generation is slow**
- Reduce resolution or FPS
- Use fewer/smaller images
- Consider using a faster codec

## Performance

Typical processing times (on modern hardware):
- Audio analysis: 5-30 seconds per song
- Video generation: 1-5 minutes per minute of video (30fps, 1080p)

## License

MIT License - Feel free to use and modify!

## Credits

Built with:
- [librosa](https://librosa.org/) - Audio analysis
- [OpenCV](https://opencv.org/) - Video processing
- [Pillow](https://python-pillow.org/) - Image manipulation
- [ffmpeg](https://ffmpeg.org/) - Audio/video encoding

---

**Made for creators who want to make viral content! 🚀**

# 🎬 ReelMaker - Project Overview

## What is ReelMaker?

ReelMaker is an automated video generation system that creates viral-style videos by synchronizing visual effects with audio beats and energy peaks. Simply provide some images and an audio file, and ReelMaker will create a professionally synchronized video perfect for TikTok, Instagram Reels, or YouTube Shorts.

## How It Works

### The Technology

1. **Audio Analysis (Music Theory)**
   - Uses `librosa` for digital signal processing
   - Detects tempo (BPM) using beat tracking algorithms
   - Calculates onset strength envelope (energy changes over time)
   - Identifies peaks using statistical methods:
     - **Small peaks** (70th percentile) → Image effects
     - **Big peaks** (90th percentile) → Image transitions

2. **Adaptive Tempo Detection**
   - Works with ANY tempo (slow ballads to fast electronic)
   - Automatically adjusts effect frequency based on BPM
   - No manual tempo input required
   - Handles tempo changes within songs

3. **Visual Effect System**
   - **Small Peak Effects**: Pulsation, hue shift, rotation, zoom, blur
   - **Big Peak Transitions**: Fade, slide, zoom-based transitions
   - Effects are applied using PIL and OpenCV
   - Smooth easing functions for natural motion

4. **Video Rendering**
   - Frame-by-frame rendering at 30 or 60 FPS
   - Synchronized audio using ffmpeg
   - Multiple resolution support
   - Efficient processing pipeline

## Project Structure

```
ReelMaker/
├── Core Scripts
│   ├── reelmaker.py          # Main CLI interface
│   ├── audio_analyzer.py     # Audio analysis engine
│   ├── video_generator.py    # Video rendering engine
│   └── split_songs.py        # Audio file splitting utility
│
├── Documentation
│   ├── README.md             # Complete documentation
│   ├── QUICKSTART.md         # Quick tutorial
│   └── PROJECT_OVERVIEW.md   # This file
│
├── Examples & Testing
│   ├── demo.py               # Interactive demo
│   ├── examples.py           # API usage examples
│   └── test_setup.py         # Setup verification
│
├── Dependencies
│   └── requirements.txt      # Python packages
│
└── Data
    └── songs/
        ├── list.txt          # Song tracklist
        ├── ViralSongs.mp3    # Combined audio (optional)
        └── individual/       # Split songs + analysis
```

## Key Features

### ✅ Completed Features

1. **Audio Processing**
   - ✅ Split large audio files by timestamp
   - ✅ Automatic tempo detection
   - ✅ Beat tracking
   - ✅ Energy peak detection (small & big)
   - ✅ JSON export of analysis data
   - ✅ Visualization of analysis

2. **Video Generation**
   - ✅ Image loading and preprocessing
   - ✅ 5 types of image effects
   - ✅ 7 types of transitions
   - ✅ Frame-by-frame rendering
   - ✅ Audio synchronization
   - ✅ Multiple resolution support

3. **User Interface**
   - ✅ Command-line interface
   - ✅ Python API
   - ✅ Batch processing
   - ✅ Progress indicators

4. **Documentation**
   - ✅ Complete README
   - ✅ Quick start guide
   - ✅ Code examples
   - ✅ Setup testing

### 🚀 Potential Enhancements

Future features that could be added:

1. **Advanced Effects**
   - 3D transformations
   - Particle effects
   - Text overlays with animation
   - Color grading presets
   - Ken Burns effect (slow zoom/pan)

2. **Smart Features**
   - AI-based image selection
   - Automatic color correction
   - Face detection for better cropping
   - Content-aware transitions
   - Mood detection from audio

3. **Performance**
   - GPU acceleration
   - Multi-threaded rendering
   - Preview mode (low-res)
   - Resume interrupted renders

4. **User Experience**
   - GUI application
   - Web interface
   - Real-time preview
   - Effect customization UI
   - Template system

5. **Output Options**
   - Direct social media upload
   - Multiple format export
   - Watermark support
   - Subtitle generation

## Usage Patterns

### Pattern 1: Single Video Creation

```bash
# 1. Analyze audio
python reelmaker.py analyze song.mp3 --visualize

# 2. Create video
python reelmaker.py create images/ song.mp3 -o output.mp4
```

### Pattern 2: Batch Processing

```bash
# 1. Split songs (if needed)
python split_songs.py

# 2. Analyze all songs
python reelmaker.py batch-analyze songs/individual/

# 3. Create videos (loop through songs)
for song in songs/individual/*.mp3; do
    python reelmaker.py create images/ "$song" -o "output_$(basename $song .mp3).mp4"
done
```

### Pattern 3: Programmatic Use

```python
from audio_analyzer import analyze_song
from video_generator import create_viral_video

# Analyze
results = analyze_song("song.mp3", visualize=True)

# Create video
create_viral_video(
    images_dir="images/",
    audio_file="song.mp3",
    output_file="output.mp4",
    resolution=(1080, 1920),
    fps=30
)
```

## Technical Details

### Audio Analysis Algorithm

1. Load audio file using librosa
2. Compute Short-Time Fourier Transform (STFT)
3. Calculate onset strength envelope
4. Apply beat tracking algorithm (dynamic programming)
5. Find local maxima in onset strength
6. Classify peaks by percentile thresholds
7. Export timestamps for effects and transitions

### Video Rendering Pipeline

1. Load and preprocess images (resize, pad)
2. Create effect/transition event timeline
3. For each frame:
   - Determine current image
   - Check for active transitions
   - Apply any active effects
   - Render to frame buffer
4. Encode video with ffmpeg
5. Merge audio and video streams

### Effect Implementation

Effects use mathematical transformations:

- **Pulse**: Scale by factor `1 + sin(progress * π) * 0.15`
- **Rotation**: Rotate by `sin(progress * π) * 5°`
- **Hue Shift**: Adjust saturation by `1 + sin(progress * π) * 0.5`
- **Zoom**: Scale with center crop
- **Blur**: Gaussian blur with radius based on progress

All effects use easing functions for smooth animation.

## Dependencies Explained

| Package | Purpose |
|---------|---------|
| librosa | Audio analysis and beat detection |
| numpy | Numerical computations |
| scipy | Signal processing (peak detection) |
| matplotlib | Visualization of analysis |
| opencv-python | Video frame manipulation |
| Pillow | Image processing and effects |
| soundfile | Audio file I/O |
| ffmpeg | Video encoding (external) |

## Performance Metrics

Typical processing times on modern hardware:

- **Audio Analysis**: 5-30 seconds per song
- **Video Rendering**: 
  - 30fps, 1080p: ~1-2 minutes per minute of video
  - 60fps, 1080p: ~2-4 minutes per minute of video

Memory usage:
- Audio analysis: ~200-500 MB
- Video rendering: ~500MB - 2GB depending on resolution

## Best Practices

1. **For Best Audio Analysis**
   - Use high-quality audio (320kbps MP3 or better)
   - Songs with clear beats work best
   - Electronic/pop music easier than classical

2. **For Best Video Quality**
   - Use high-resolution images (1080p+)
   - Keep consistent aspect ratios
   - Use good lighting in photos
   - Mix wide and close shots

3. **For Best Performance**
   - Use SSD for faster I/O
   - Close other applications during rendering
   - Use lower resolution for previews
   - Consider reducing FPS for faster iteration

4. **For Viral Content**
   - Match image mood to music
   - Use trending songs
   - Keep videos 15-60 seconds
   - Export at platform-specific resolutions

## Platform-Specific Formats

| Platform | Resolution | Aspect Ratio | Duration |
|----------|-----------|--------------|----------|
| TikTok | 1080x1920 | 9:16 | 15-60s |
| Instagram Reels | 1080x1920 | 9:16 | 15-90s |
| Instagram Feed | 1080x1080 | 1:1 | Any |
| YouTube Shorts | 1080x1920 | 9:16 | 15-60s |
| YouTube Regular | 1920x1080 | 16:9 | Any |

## Troubleshooting Guide

See README.md for detailed troubleshooting.

Quick fixes:
- **Slow performance**: Reduce resolution or FPS
- **Off-beat effects**: Re-analyze audio
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **ffmpeg errors**: Reinstall ffmpeg

## Credits & License

**Built with:**
- Python 3.8+
- Open source libraries (see requirements.txt)

**License:** MIT

**Created for:** Content creators who want to automate viral video creation

---

**Get Started:** Run `python test_setup.py` to verify your installation!

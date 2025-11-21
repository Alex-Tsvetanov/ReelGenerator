# 🎬 ReelMaker

> **Create viral videos by synchronizing images with audio beats and energy peaks**

Transform your photos and music into engaging videos perfect for TikTok, Instagram Reels, and YouTube Shorts - automatically!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- 🎵 **Intelligent Audio Analysis** - Automatic tempo detection and beat tracking
- 🎨 **Dynamic Visual Effects** - Pulsation, rotation, hue shifts, and more
- 🔄 **Smooth Transitions** - Fade, slide, and zoom between images
- 🎯 **Beat-Perfect Timing** - Effects sync perfectly with music energy peaks
- 📱 **Multi-Platform** - Export for TikTok, Reels, Shorts, or any resolution
- ⚡ **Fast Processing** - Efficient rendering pipeline
- 🎛️ **Customizable** - Full Python API for advanced users

## 🎥 How It Works

```
Your Images + Your Music → AI Analysis → Synchronized Video
```

1. **Audio Analysis**: Detects tempo, beats, and energy peaks
2. **Effect Planning**: Maps peaks to visual effects and transitions
3. **Video Generation**: Renders synchronized video at your chosen resolution

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ReelMaker.git
cd ReelMaker

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (if not already installed)
# Windows: choco install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Verify setup
python test_setup.py
```

### Basic Usage

```bash
# 1. Analyze your audio
python reelmaker.py analyze song.mp3 --visualize

# 2. Create your video
python reelmaker.py create images/ song.mp3 -o output.mp4
```

That's it! You now have a viral-ready video.

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Step-by-step tutorial
- **[Full Documentation](README.md)** - Complete feature reference
- **[Project Overview](PROJECT_OVERVIEW.md)** - Technical details
- **[Build Summary](BUILD_SUMMARY.md)** - What's included

## 💡 Examples

### Portrait Video (TikTok/Reels)
```bash
python reelmaker.py create images/ song.mp3 --resolution 1080x1920 -o reel.mp4
```

### Landscape Video (YouTube)
```bash
python reelmaker.py create images/ song.mp3 --resolution 1920x1080 -o video.mp4
```

### High FPS (Smooth Effects)
```bash
python reelmaker.py create images/ song.mp3 --fps 60 -o smooth.mp4
```

### Batch Process Multiple Songs
```bash
python reelmaker.py batch-analyze songs/
```

## 🎨 Visual Effects

### Small Peak Effects (Beat Accents)
- 💫 Pulse/Scale
- 🌈 Hue Shift
- 🔄 Rotation
- 🔍 Zoom
- 💨 Blur Pulse

### Big Peak Transitions (Major Beats)
- 🌅 Fade
- ⬅️➡️ Slide (all directions)
- 🎯 Zoom In/Out

## 🎵 Music Support

Works with **any music genre**:
- ✅ Electronic/EDM (fast BPM)
- ✅ Pop music (medium BPM)
- ✅ Rock/Metal (variable BPM)
- ✅ Hip-Hop/Rap (rhythmic)
- ✅ Classical (complex timing)
- ✅ Ballads (slow BPM)

The system adapts automatically to the tempo!

## 📱 Platform Support

| Platform | Resolution | Command |
|----------|-----------|---------|
| TikTok | 1080x1920 | `--resolution 1080x1920` |
| Instagram Reels | 1080x1920 | `--resolution 1080x1920` |
| Instagram Feed | 1080x1080 | `--resolution 1080x1080` |
| YouTube Shorts | 1080x1920 | `--resolution 1080x1920` |
| YouTube | 1920x1080 | `--resolution 1920x1080` |

## 🔧 Python API

```python
from audio_analyzer import analyze_song
from video_generator import create_viral_video

# Analyze audio
results = analyze_song("song.mp3", visualize=True)
print(f"Detected {results['num_big_peaks']} transitions")

# Create video
create_viral_video(
    images_dir="images/",
    audio_file="song.mp3",
    output_file="output.mp4",
    resolution=(1080, 1920),
    fps=30
)
```

See [examples.py](examples.py) for more!

## 🎯 Use Cases

- 📸 **Photo Montages** - Turn memories into engaging videos
- 🎤 **Music Visualizers** - Create beat-synced visual content
- 📱 **Social Media** - Generate viral TikToks and Reels
- 🎉 **Event Recaps** - Wedding, party, or travel videos
- 📢 **Marketing** - Eye-catching promotional content
- 🎨 **Art Projects** - Creative audio-visual experiments

## ⚡ Performance

- **Audio Analysis**: 5-30 seconds per song
- **Video Rendering**: 1-5 minutes per minute of video (30fps, 1080p)
- **Memory Usage**: 500MB - 2GB

## 🛠️ Requirements

- Python 3.8+
- ffmpeg
- ~2GB RAM minimum
- Works on Windows, Mac, Linux

## 📦 What's Included

- `reelmaker.py` - Main CLI interface
- `audio_analyzer.py` - Beat detection engine
- `video_generator.py` - Video rendering engine
- `split_songs.py` - Audio file splitter
- Complete documentation
- Example scripts
- Test utilities

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your creations

## 📄 License

MIT License - feel free to use this for personal or commercial projects!

## 🙏 Credits

Built with:
- [librosa](https://librosa.org/) - Audio analysis
- [OpenCV](https://opencv.org/) - Video processing
- [Pillow](https://python-pillow.org/) - Image manipulation
- [NumPy](https://numpy.org/) - Numerical computing
- [ffmpeg](https://ffmpeg.org/) - Media encoding

## 🌟 Star This Project

If you find ReelMaker useful, please give it a star! It helps others discover it.

## 📞 Support

- 📖 Check the [documentation](README.md)
- 🐛 [Report issues](https://github.com/yourusername/ReelMaker/issues)
- 💬 [Discussions](https://github.com/yourusername/ReelMaker/discussions)

---

**Made with ❤️ for content creators**

Start creating viral videos today! 🚀

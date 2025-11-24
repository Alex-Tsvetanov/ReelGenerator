# ReelMaker AI Coding Instructions

## Project Overview
ReelMaker is a Python-based viral video generator that synchronizes visual effects with audio beats and energy peaks. It uses music theory (tempo detection, beat tracking, onset strength) to create TikTok/Instagram Reels-style videos from static images.

## Architecture & Data Flow

### Three-Module Pipeline
1. **Audio Analysis** (`audio_analyzer.py`) → JSON analysis files
2. **Video Generation** (`video_generator.py`) → MP4 output
3. **CLI Interface** (`reelmaker.py`) → User commands

**Critical Data Flow:**
```
Audio file → analyze_song() → {song}_analysis.json
                                      ↓
Images dir + audio + analysis.json → create_viral_video() → output.mp4
```

**Analysis JSON Structure:**
```json
{
  "tempo_bpm": float,
  "beat_times": [float],          // All detected beats
  "small_peak_times": [float],    // Trigger effects (pulse, hue shift, rotate)
  "big_peak_times": [float]       // Trigger transitions (fade, slide)
}
```

### Key Design Decisions

**Two-stage processing:** Audio analysis is deliberately separated from video generation to enable:
- Batch video creation with multiple image sets
- Quick iteration on visual styles without re-analyzing audio
- Pre-analysis of large song libraries (see `batch_gen.sh` pattern)

**Peak detection thresholds:** Uses percentile-based classification (70th/90th by default in `detect_energy_peaks()`). Small peaks drive image effects; big peaks drive transitions between images. This separation prevents visual chaos.

**Minimum transition interval:** Big peaks from audio analysis can cluster too closely, causing disorienting rapid transitions. The `min_transition_interval` parameter (default 2.0s) filters out transitions that occur too soon after the previous one. See `plan_effects_and_transitions()` in `video_generator.py` - implements greedy filtering where transitions are kept only if they're at least N seconds after the last kept transition.

**Image cycling:** Videos cycle through available images using modulo arithmetic (`i % num_images`). Fewer images = repetition, which can actually enhance viral appeal.

## Critical Workflows

### Batch Video Generation
```bash
# 1. Batch analyze all songs
python reelmaker.py batch-analyze songs/individual/

# 2. Generate videos for all analyses
for analysis in songs/individual/analysis/*_analysis.json; do
    song_name=$(basename "$analysis" _analysis.json)
    python reelmaker.py create images/ "songs/individual/$song_name.mp3" \
        --analysis "$analysis" -o "output/$song_name.mp4"
done
```

### Testing New Effects/Transitions
When adding effects to `VideoGenerator`, test with `--max-duration 10` flag to render only first 10 seconds during development. See `examples.py` for programmatic API usage.

## Project-Specific Conventions

### Image Processing Patterns
**Letterbox handling** (`video_generator.py` lines 250-280): Images are fitted to width with blurred backgrounds. The `padding_mask` (boolean array) tracks letterbox areas to prevent effect artifacts on transparent regions. When applying effects (rotation, zoom), both the image AND mask must be transformed identically.

**EXIF orientation:** Always call `ImageOps.exif_transpose()` after loading images to fix phone photo rotations (see `load_images()` method).

### Audio Analysis Patterns
**Adaptive tempo detection:** The system handles any BPM (40-200+). Peak detection uses `min_distance` based on tempo to prevent over-detection in fast songs. See `detect_energy_peaks()` lines 110-120.

**Onset strength vs spectral energy:** `librosa.onset.onset_strength()` detects rhythmic events; `librosa.feature.rms()` measures overall energy. Both are used together for robust peak detection.

### Video Rendering
**Frame-by-frame rendering:** Each frame is composed by:
1. Determining active transition (if any) based on current time
2. Applying active effects to current image
3. Compositing foreground on blurred background
4. Converting PIL → numpy → OpenCV format

**ffmpeg synchronization:** Video frames are rendered first, then audio is merged using ffmpeg's `-i audio.mp3 -shortest` pattern. See `_combine_with_audio()` method.

## External Dependencies

### Required System Tools
- **ffmpeg**: Must be in PATH. Used for audio merging and format conversion. Verify with `test_setup.py`
- **librosa**: Core audio analysis. Requires `soundfile` backend on Windows

### Common Integration Issues
- Large images (>4000px): Auto-downscaled in `load_images()` to prevent OOM errors
- Slow analysis: Normal for long songs (2-5 min). Use `batch-analyze` overnight for libraries
- Black frames on rotation: Indicates missing alpha channel handling in effect composition

## File Organization

### Key Directories
- `songs/individual/`: Split audio files + per-song analysis JSONs
- `songs/individual/analysis/`: Alternate analysis output location (used by `batch-analyze`)
- `images/`: Input image directory (user-provided)
- `output/`: Generated videos

### Analysis File Locations
Analysis JSONs are saved adjacent to audio files by default (`{audio_stem}_analysis.json`). The `--analysis` CLI flag allows override for custom analysis storage.

## Testing & Debugging

### Quick Verification
```bash
python test_setup.py  # Verify all dependencies
python demo.py        # Run complete workflow demo
```

### Analysis Visualization
Always use `--visualize` flag during development to generate PNG plots showing waveform, onset strength, detected peaks, and spectrogram. These reveal rhythm detection issues instantly.

### Common Debugging Patterns
- **No transitions detected:** Check if `big_peak_percentile` is too high; reduce to 85
- **Too many effects:** Increase `small_peak_percentile` to 75-80
- **Misaligned beats:** Indicates complex tempo changes; librosa's beat tracker may need manual tempo hint
- **Memory errors:** Large image count or high resolution; use `--max-duration` for testing

## Command Examples

```bash
# Portrait video (TikTok/Reels)
python reelmaker.py create images/ song.mp3 --resolution 1080x1920 --fps 60

# Landscape video (YouTube Shorts)
python reelmaker.py create images/ song.mp3 --resolution 1920x1080

# Test render (first 10 seconds only)
python reelmaker.py create images/ song.mp3 --max-duration 10 -o test.mp4

# Use pre-computed analysis from different location
python reelmaker.py create images/ song.mp3 --analysis path/to/custom_analysis.json

# Adjust minimum transition interval for slower/faster pacing
python reelmaker.py create images/ song.mp3 --min-transition-interval 3.0  # Slower (3s per image min)
python reelmaker.py create images/ song.mp3 --min-transition-interval 1.5  # Faster pacing
```

## Code Style Notes
- Type hints used throughout for clarity (see function signatures)
- Numpy/PIL/OpenCV interop requires careful array format management (see `sanitize_frame()`)
- Progress indicators use simple print statements (no fancy progress bars)
- Dataclasses (`EffectEvent`, `TransitionEvent`) structure event planning

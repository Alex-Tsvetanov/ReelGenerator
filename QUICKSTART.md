# Quick Start Guide

## Step-by-Step Tutorial

### Step 1: Verify Setup
```bash
python test_setup.py
```
Should show all ✓ checks passing.

### Step 2: Analyze Your First Song

Pick any song from the `songs/individual/` directory (or use your own):

```bash
python reelmaker.py analyze "songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3" --visualize
```

This creates:
- `ANDROMEDA, elysian. - MONTAGEM COMA_analysis.json` - Data file
- `ANDROMEDA, elysian. - MONTAGEM COMA_analysis.png` - Visualization

**What to look for:**
- Check the "Big peaks (transitions)" number
- You'll need roughly that many images for your video

### Step 3: Prepare Your Images

Create a folder and add images:

```bash
mkdir my_images
# Copy 10-20 images into my_images/
```

**Image tips:**
- Use high-quality JPG or PNG
- Mix different types (portraits, landscapes, close-ups)
- More images = more variety
- Fewer images = images will repeat (which can look cool!)

### Step 4: Create Your Video

```bash
python reelmaker.py create my_images/ "songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3" -o my_first_reel.mp4
```

**What happens:**
1. Loads your images
2. Reads the audio analysis
3. Plans effects and transitions
4. Renders each frame with synchronized effects
5. Combines with audio
6. Saves final video

**This will take 2-5 minutes depending on your computer.**

### Step 5: Watch and Share!

Open `my_first_reel.mp4` and watch your creation!

## Common Commands

### Portrait Video (TikTok/Reels style)
```bash
python reelmaker.py create images/ song.mp3 --resolution 1080x1920 -o vertical.mp4
```

### Landscape Video (YouTube)
```bash
python reelmaker.py create images/ song.mp3 --resolution 1920x1080 -o horizontal.mp4
```

### Square Video (Instagram Feed)
```bash
python reelmaker.py create images/ song.mp3 --resolution 1080x1080 -o square.mp4
```

### High Frame Rate (Smoother)
```bash
python reelmaker.py create images/ song.mp3 --fps 60 -o smooth.mp4
```

### Analyze All Songs at Once
```bash
python reelmaker.py batch-analyze songs/individual/
```

## Understanding the Analysis

When you analyze a song, you'll see output like:

```
Tempo: 117.5 BPM
Beats: 145
Small peaks (effects): 51
Big peaks (transitions): 26
```

**What this means:**
- **Tempo**: Song speed in beats per minute
- **Beats**: Total number of beats detected
- **Small peaks**: Times when effects happen (pulsation, rotation, etc.)
- **Big peaks**: Times when transitions happen (change images)

**For this example:**
- The song has 26 transition points
- So you need ~26-30 images for best results
- Effects happen every ~1.4 seconds
- Transitions happen every ~2.8 seconds

## Troubleshooting

### "No images found"
Make sure you have .jpg, .png, or .jpeg files in your images folder.

### "Analysis file not found"
Run the analyze command first before creating a video.

### Video looks off-beat
Some songs have irregular rhythms. Try:
1. Re-analyzing with a different song
2. Using songs with clear, steady beats (electronic music works great!)

### Video generation is slow
Normal! Video rendering takes time. For a 3-minute song:
- At 30fps: ~3-4 minutes to render
- At 60fps: ~6-8 minutes to render

Lower resolution or fps if you need faster previews.

## Tips for Viral Content

1. **Choose energetic songs** - High BPM (100+) works best
2. **Use eye-catching images** - Bold colors, interesting subjects
3. **Mix portrait and landscape** - Variety keeps viewers engaged
4. **Test different effects** - The algorithm picks effects automatically
5. **Match content to music** - Upbeat music → bright images, slow music → moody images

## Next Steps

Once you're comfortable:
- Experiment with different songs and image sets
- Try different resolutions for different platforms
- Batch process multiple videos
- Check out the code to customize effects

**Happy creating! 🎬✨**

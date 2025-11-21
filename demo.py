"""
Demo script to showcase ReelMaker capabilities.

This script demonstrates the complete workflow:
1. Analyze an audio file
2. Create a sample video with the analyzed audio
"""

import sys
from pathlib import Path
from audio_analyzer import analyze_song
from video_generator import create_viral_video


def demo_analysis():
    """Demo: Analyze an audio file."""
    print("\n" + "=" * 70)
    print("DEMO 1: AUDIO ANALYSIS")
    print("=" * 70)
    
    # Use one of the split songs
    songs_dir = Path("songs/individual")
    
    if not songs_dir.exists():
        print("Error: songs/individual directory not found.")
        print("Please run split_songs.py first to split the audio file.")
        return None
    
    # Get first song
    audio_files = list(songs_dir.glob("*.mp3"))
    if not audio_files:
        print("Error: No MP3 files found in songs/individual/")
        return None
    
    demo_song = audio_files[0]
    print(f"\nAnalyzing: {demo_song.name}")
    
    # Analyze
    results = analyze_song(str(demo_song), visualize=True, save_json=True)
    
    print("\n" + "-" * 70)
    print("ANALYSIS RESULTS:")
    print("-" * 70)
    print(f"Duration:        {results['duration']:.2f}s")
    print(f"Tempo:           {results['tempo_bpm']:.1f} BPM")
    print(f"Total beats:     {results['num_beats']}")
    print(f"Small peaks:     {results['num_small_peaks']} (for effects)")
    print(f"Big peaks:       {results['num_big_peaks']} (for transitions)")
    print(f"\nRecommendation:  Use {results['num_big_peaks'] + 1} to {results['num_big_peaks'] + 5} images")
    print(f"Effect interval: ~{results['duration']/results['num_small_peaks']:.2f}s")
    print(f"Transition rate: ~{results['duration']/results['num_big_peaks']:.2f}s")
    
    return str(demo_song)


def demo_video_creation(audio_file: str = None):
    """Demo: Create a video from images and audio."""
    print("\n" + "=" * 70)
    print("DEMO 2: VIDEO CREATION")
    print("=" * 70)
    
    # Check for demo images
    demo_images_dir = Path("demo_images")
    
    if not demo_images_dir.exists():
        print("\n⚠️  Demo images directory not found.")
        print("To run the video generation demo:")
        print("1. Create a 'demo_images' directory")
        print("2. Add 10-20 images to it")
        print("3. Run this demo again")
        print("\nAlternatively, use the CLI:")
        print("  python reelmaker.py create <your_images_dir> <audio_file>")
        return
    
    # Count images
    image_files = list(demo_images_dir.glob("*.jpg")) + \
                 list(demo_images_dir.glob("*.png")) + \
                 list(demo_images_dir.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"\n⚠️  No images found in {demo_images_dir}")
        print("Please add some images and run again.")
        return
    
    print(f"\nFound {len(image_files)} images in {demo_images_dir}")
    
    # Use provided audio or find one
    if audio_file is None:
        songs_dir = Path("songs/individual")
        audio_files = list(songs_dir.glob("*.mp3"))
        if not audio_files:
            print("Error: No audio files found")
            return
        audio_file = str(audio_files[0])
    
    audio_path = Path(audio_file)
    print(f"Using audio: {audio_path.name}")
    
    # Check for analysis
    analysis_file = audio_path.parent / f"{audio_path.stem}_analysis.json"
    if not analysis_file.exists():
        print(f"\n⚠️  Analysis file not found: {analysis_file.name}")
        print("Analyzing audio first...")
        analyze_song(audio_file, visualize=False, save_json=True)
    
    # Create video
    output_file = "demo_output.mp4"
    print(f"\nGenerating video: {output_file}")
    print("This may take a few minutes...\n")
    
    try:
        create_viral_video(
            images_dir=str(demo_images_dir),
            audio_file=audio_file,
            analysis_file=str(analysis_file),
            output_file=output_file,
            resolution=(1080, 1920),  # Portrait
            fps=30
        )
        
        print("\n" + "=" * 70)
        print("✓ DEMO VIDEO CREATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Output: {output_file}")
        print("\nYou can now:")
        print("1. Play the video to see the synchronized effects")
        print("2. Upload to TikTok, Instagram Reels, or YouTube Shorts")
        print("3. Adjust settings and regenerate with different parameters")
        
    except Exception as e:
        print(f"\n✗ Error creating video: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the demo."""
    print("=" * 70)
    print("REELMAKER DEMO")
    print("=" * 70)
    print("\nThis demo will show you:")
    print("1. How to analyze audio for beats and peaks")
    print("2. How to create a synchronized video (if demo images are provided)")
    
    # Demo 1: Audio Analysis
    audio_file = demo_analysis()
    
    # Demo 2: Video Creation
    if audio_file:
        demo_video_creation(audio_file)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check the generated analysis files (JSON and PNG)")
    print("2. If you created demo images, check demo_output.mp4")
    print("3. Read README.md for full usage instructions")
    print("4. Use reelmaker.py for the full CLI experience")
    print("\nHappy creating! 🎬✨")


if __name__ == '__main__':
    main()

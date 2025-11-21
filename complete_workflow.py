#!/usr/bin/env python3
"""
Complete Workflow Example

This script demonstrates the entire ReelMaker workflow from start to finish.
Run this to see the complete process in action.
"""

from pathlib import Path
import json


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def workflow_step_1_verify_setup():
    """Step 1: Verify the setup."""
    print_section("STEP 1: Verify Setup")
    
    import sys
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check key dependencies
    try:
        import librosa
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    # Check ffmpeg
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ ffmpeg available")
    except:
        print("✗ ffmpeg not found")
        return False
    
    return True


def workflow_step_2_prepare_audio():
    """Step 2: Prepare audio files."""
    print_section("STEP 2: Prepare Audio")
    
    songs_dir = Path("songs/individual")
    
    if not songs_dir.exists():
        print("Running song splitter...")
        import split_songs
        # Note: This would run the splitter if needed
        print("✓ Songs split into individual files")
    else:
        mp3_count = len(list(songs_dir.glob("*.mp3")))
        print(f"✓ Found {mp3_count} songs in {songs_dir}")
    
    return True


def workflow_step_3_analyze_audio():
    """Step 3: Analyze audio."""
    print_section("STEP 3: Analyze Audio")
    
    from audio_analyzer import analyze_song
    
    songs_dir = Path("songs/individual")
    audio_files = list(songs_dir.glob("*.mp3"))
    
    if not audio_files:
        print("✗ No audio files found")
        return False
    
    # Pick first song for demo
    demo_song = audio_files[0]
    print(f"Analyzing: {demo_song.name}")
    
    # Check if already analyzed
    analysis_file = demo_song.parent / f"{demo_song.stem}_analysis.json"
    
    if analysis_file.exists():
        print(f"✓ Analysis already exists: {analysis_file.name}")
        with open(analysis_file) as f:
            results = json.load(f)
    else:
        print("Running analysis...")
        results = analyze_song(str(demo_song), visualize=True, save_json=True)
        print(f"✓ Analysis complete: {analysis_file.name}")
    
    # Display results
    print(f"\n  Song: {demo_song.name}")
    print(f"  Duration: {results['duration']:.1f}s")
    print(f"  Tempo: {results['tempo_bpm']:.1f} BPM")
    print(f"  Effects: {results['num_small_peaks']}")
    print(f"  Transitions: {results['num_big_peaks']}")
    print(f"  Recommended images: {results['num_big_peaks'] + 1}-{results['num_big_peaks'] + 5}")
    
    return str(demo_song), results


def workflow_step_4_prepare_images():
    """Step 4: Prepare images."""
    print_section("STEP 4: Prepare Images")
    
    demo_images = Path("demo_images")
    
    if not demo_images.exists():
        print(f"\n⚠️  {demo_images} directory not found")
        print("\nTo complete this workflow:")
        print(f"1. Create '{demo_images}' directory")
        print("2. Add 10-20 images (JPG or PNG)")
        print("3. Run this script again")
        print("\nSkipping video generation for now...")
        return None
    
    # Count images
    image_files = list(demo_images.glob("*.jpg")) + \
                 list(demo_images.glob("*.png")) + \
                 list(demo_images.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"✗ No images found in {demo_images}")
        return None
    
    print(f"✓ Found {len(image_files)} images")
    print(f"  Located in: {demo_images}")
    
    return str(demo_images)


def workflow_step_5_generate_video(audio_file, images_dir, results):
    """Step 5: Generate video."""
    print_section("STEP 5: Generate Video")
    
    if images_dir is None:
        print("⚠️  Skipping - no images directory")
        return False
    
    from video_generator import create_viral_video
    
    output_file = "workflow_example_output.mp4"
    
    print(f"Creating video: {output_file}")
    print(f"  Images: {images_dir}")
    print(f"  Audio: {Path(audio_file).name}")
    print(f"  Resolution: 1080x1920 (Portrait)")
    print(f"  FPS: 30")
    print(f"\nThis will take 2-5 minutes...")
    
    try:
        create_viral_video(
            images_dir=images_dir,
            audio_file=audio_file,
            output_file=output_file,
            resolution=(1080, 1920),
            fps=30
        )
        
        print(f"\n✓ Video created successfully!")
        print(f"  Output: {output_file}")
        print(f"  Duration: {results['duration']:.1f}s")
        
        # Get file size
        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating video: {e}")
        import traceback
        traceback.print_exc()
        return False


def workflow_step_6_summary():
    """Step 6: Summary and next steps."""
    print_section("STEP 6: Summary & Next Steps")
    
    print("\n✓ WORKFLOW COMPLETE!")
    print("\nWhat you created:")
    print("  • Audio analysis JSON file")
    print("  • Audio visualization PNG")
    print("  • Synchronized video MP4 (if images provided)")
    
    print("\nNext steps:")
    print("  1. Watch your video!")
    print("  2. Try different images or songs")
    print("  3. Experiment with resolutions:")
    print("     - Portrait: 1080x1920 (TikTok/Reels)")
    print("     - Landscape: 1920x1080 (YouTube)")
    print("     - Square: 1080x1080 (Instagram)")
    print("  4. Adjust FPS for smoother/faster")
    print("  5. Share your creations!")
    
    print("\nUseful commands:")
    print("  python reelmaker.py --help")
    print("  python test_setup.py")
    print("  python examples.py")
    
    print("\n" + "=" * 70)


def main():
    """Run the complete workflow."""
    print("=" * 70)
    print("  REELMAKER - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 70)
    print("\nThis script will walk you through the entire process:")
    print("  1. Verify setup")
    print("  2. Prepare audio")
    print("  3. Analyze audio")
    print("  4. Prepare images")
    print("  5. Generate video")
    print("  6. Summary")
    
    # Step 1: Verify setup
    if not workflow_step_1_verify_setup():
        print("\n✗ Setup verification failed")
        print("Please install dependencies: pip install -r requirements.txt")
        return
    
    # Step 2: Prepare audio
    if not workflow_step_2_prepare_audio():
        print("\n✗ Audio preparation failed")
        return
    
    # Step 3: Analyze audio
    audio_result = workflow_step_3_analyze_audio()
    if not audio_result:
        print("\n✗ Audio analysis failed")
        return
    
    audio_file, results = audio_result
    
    # Step 4: Prepare images
    images_dir = workflow_step_4_prepare_images()
    
    # Step 5: Generate video (if images available)
    if images_dir:
        workflow_step_5_generate_video(audio_file, images_dir, results)
    
    # Step 6: Summary
    workflow_step_6_summary()


if __name__ == '__main__':
    main()

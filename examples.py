"""
Example: Using ReelMaker as a Python library

This example shows how to use ReelMaker programmatically
instead of using the command-line interface.
"""

from pathlib import Path
from audio_analyzer import AudioAnalyzer, analyze_song
from video_generator import VideoGenerator, create_viral_video


def example_1_basic_analysis():
    """Example 1: Basic audio analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Audio Analysis")
    print("=" * 70)
    
    # Path to your audio file
    audio_file = "songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3"
    
    # Simple way: Use convenience function
    results = analyze_song(audio_file, visualize=True, save_json=True)
    
    print(f"\nTempo: {results['tempo_bpm']:.1f} BPM")
    print(f"Duration: {results['duration']:.1f}s")
    print(f"Transitions needed: {results['num_big_peaks']}")


def example_2_custom_analysis():
    """Example 2: Custom audio analysis with fine control."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Audio Analysis")
    print("=" * 70)
    
    audio_file = "songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3"
    
    # Create analyzer object
    analyzer = AudioAnalyzer(audio_file)
    
    # Load audio
    analyzer.load_audio()
    
    # Analyze tempo and beats
    tempo, beat_times = analyzer.analyze_tempo_and_beats()
    print(f"\nDetected tempo: {tempo:.1f} BPM")
    print(f"Found {len(beat_times)} beats")
    
    # Detect peaks with custom thresholds
    # Higher percentile = fewer peaks (more selective)
    small_peaks, big_peaks = analyzer.detect_energy_peaks(
        small_peak_percentile=75,  # Top 25% for effects
        big_peak_percentile=92     # Top 8% for transitions
    )
    
    print(f"Small peaks: {len(small_peaks)}")
    print(f"Big peaks: {len(big_peaks)}")
    
    # Access raw data
    print(f"\nFirst 5 beat times: {beat_times[:5].tolist()}")
    print(f"First 3 transition times: {big_peaks[:3].tolist()}")


def example_3_basic_video():
    """Example 3: Basic video generation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Basic Video Generation")
    print("=" * 70)
    
    # Check if we have demo images
    if not Path("demo_images").exists():
        print("\n⚠️  Create 'demo_images' folder with images first")
        return
    
    # Simple way: Use convenience function
    create_viral_video(
        images_dir="demo_images",
        audio_file="songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3",
        output_file="example_output.mp4",
        resolution=(1080, 1920),  # Portrait
        fps=30
    )
    
    print("\n✓ Video created: example_output.mp4")


def example_4_custom_video():
    """Example 4: Custom video generation with full control."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Video Generation")
    print("=" * 70)
    
    # Check if we have demo images
    if not Path("demo_images").exists():
        print("\n⚠️  Create 'demo_images' folder with images first")
        return
    
    # Create generator object
    generator = VideoGenerator(
        images_dir="demo_images",
        audio_file="songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA.mp3",
        analysis_file="songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA_analysis.json",
        output_file="custom_output.mp4",
        resolution=(720, 1280),  # Smaller resolution for faster rendering
        fps=24  # Cinematic frame rate
    )
    
    # Load everything
    generator.load_images()
    generator.load_analysis()
    
    # Plan effects and transitions
    generator.plan_effects_and_transitions()
    
    # Customize effects before generation (optional)
    print(f"\nPlanned {len(generator.effect_events)} effects")
    print(f"Planned {len(generator.transition_events)} transitions")
    
    # You could modify generator.effect_events or generator.transition_events here
    # For example, to change effect durations:
    for effect in generator.effect_events:
        effect.duration = 0.2  # Make effects last longer
        effect.intensity = 0.9  # Make effects stronger
    
    # Generate the video
    generator.generate_video()
    
    print("\n✓ Custom video created: custom_output.mp4")


def example_5_batch_processing():
    """Example 5: Batch process multiple songs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 70)
    
    songs_dir = Path("songs/individual")
    
    # Get all MP3 files
    audio_files = list(songs_dir.glob("*.mp3"))[:3]  # Just first 3 for demo
    
    print(f"\nProcessing {len(audio_files)} songs...\n")
    
    for audio_file in audio_files:
        print(f"\n{'='*50}")
        print(f"Processing: {audio_file.name}")
        print(f"{'='*50}")
        
        # Analyze
        results = analyze_song(str(audio_file), visualize=False, save_json=True)
        
        print(f"  Tempo: {results['tempo_bpm']:.1f} BPM")
        print(f"  Duration: {results['duration']:.1f}s")
        print(f"  Transitions: {results['num_big_peaks']}")
        
        # You could create videos here too if you have images
        # create_viral_video(
        #     images_dir=f"images_{audio_file.stem}",
        #     audio_file=str(audio_file),
        #     output_file=f"output_{audio_file.stem}.mp4"
        # )


def example_6_accessing_peak_data():
    """Example 6: Access and use peak timing data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Working with Peak Data")
    print("=" * 70)
    
    import json
    
    # Load analysis from JSON
    analysis_file = "songs/individual/ANDROMEDA, elysian. - MONTAGEM COMA_analysis.json"
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nSong: {Path(data['audio_file']).name}")
    print(f"Duration: {data['duration']:.2f}s")
    print(f"Tempo: {data['tempo_bpm']:.1f} BPM")
    
    # Get peak times
    small_peaks = data['small_peak_times']
    big_peaks = data['big_peak_times']
    
    print(f"\nEffect peaks (first 10):")
    for i, time in enumerate(small_peaks[:10], 1):
        print(f"  {i}. Effect at {time:.2f}s")
    
    print(f"\nTransition peaks (all):")
    for i, time in enumerate(big_peaks, 1):
        print(f"  {i}. Transition at {time:.2f}s")
    
    # Calculate statistics
    import numpy as np
    if len(big_peaks) > 1:
        avg_gap = np.mean(np.diff(big_peaks))
        print(f"\nAverage time between transitions: {avg_gap:.2f}s")
        print(f"Recommended number of images: {len(big_peaks) + 1}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("REELMAKER PYTHON API EXAMPLES")
    print("=" * 70)
    print("\nThese examples show how to use ReelMaker programmatically.")
    print("You can run individual examples by importing this file.")
    
    # Check if songs exist
    if not Path("songs/individual").exists():
        print("\n⚠️  No songs found. Run split_songs.py first.")
        return
    
    # Run examples that don't require images
    example_1_basic_analysis()
    example_2_custom_analysis()
    example_5_batch_processing()
    example_6_accessing_peak_data()
    
    # Examples that need images
    if Path("demo_images").exists():
        example_3_basic_video()
        example_4_custom_video()
    else:
        print("\n" + "=" * 70)
        print("SKIPPING VIDEO EXAMPLES")
        print("=" * 70)
        print("\nTo run video generation examples:")
        print("1. Create a 'demo_images' folder")
        print("2. Add some images to it")
        print("3. Run this script again")
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nYou can now:")
    print("1. Import these functions in your own scripts")
    print("2. Modify them for your specific needs")
    print("3. Build your own video generation pipelines")


if __name__ == '__main__':
    main()

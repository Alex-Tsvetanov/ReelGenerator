"""
Main script for the Viral Video Maker - ReelMaker

This script provides a simple interface to:
1. Analyze audio files for rhythm and energy peaks
2. Generate viral-style videos from images and audio
"""

import argparse
from pathlib import Path
import sys

from audio_analyzer import analyze_song, batch_analyze_songs
from video_generator import create_viral_video


def analyze_audio_command(args):
    """Handle audio analysis command."""
    audio_path = Path(args.audio)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("AUDIO ANALYSIS")
    print("=" * 70)
    
    results = analyze_song(
        str(audio_path),
        visualize=args.visualize,
        save_json=True
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Duration: {results['duration']:.2f}s")
    print(f"Tempo: {results['tempo_bpm']:.1f} BPM")
    print(f"Beats: {results['num_beats']}")
    
    # Show summary for each detection method
    methods = [('fullspectrum', 'Full Spectrum'), 
               ('percussive', 'Percussive'), 
               ('lowfreq', 'Low-Frequency')]
    
    # Add Demucs if available
    if 'demucs' in results:
        methods.append(('demucs', 'Demucs (Drums Only)'))
    
    for method_name, method_label in methods:
        method_data = results[method_name]
        num_small = len(method_data['small_peak_times'])
        num_big = len(method_data['big_peak_times'])
        print(f"\n{method_label}:")
        print(f"  Small peaks (effects): {num_small}")
        print(f"  Big peaks (transitions): {num_big}")
        if num_big > 0:
            print(f"  Avg images needed: {num_big + 1}")
            print(f"  Effect frequency: ~{results['duration']/num_small:.2f}s")
            print(f"  Transition frequency: ~{results['duration']/num_big:.2f}s")


def batch_analyze_command(args):
    """Handle batch analysis command."""
    songs_dir = Path(args.directory)
    
    if not songs_dir.exists():
        print(f"Error: Directory not found: {songs_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("BATCH AUDIO ANALYSIS")
    print("=" * 70)
    
    output_dir = args.output if args.output else songs_dir / 'analysis'
    batch_analyze_songs(str(songs_dir), str(output_dir))


def create_video_command(args):
    """Handle video creation command."""
    images_dir = Path(args.images)
    audio_file = Path(args.audio)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Check for analysis file
    analysis_file = args.analysis
    if not analysis_file:
        analysis_file = audio_file.parent / f"{audio_file.stem}_analysis.json"
        if not analysis_file.exists():
            print(f"Error: Analysis file not found: {analysis_file}")
            print("Please run audio analysis first with: python reelmaker.py analyze <audio_file>")
            sys.exit(1)
    
    print("=" * 70)
    print("VIDEO GENERATION")
    print("=" * 70)
    
    # Parse resolution
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except:
            print(f"Error: Invalid resolution format. Use WIDTHxHEIGHT (e.g., 1080x1920)")
            sys.exit(1)
    else:
        resolution = (1080, 1920)  # Default portrait
    
    output_file = args.output if args.output else "output.mp4"
    
    create_viral_video(
        images_dir=str(images_dir),
        audio_file=str(audio_file),
        analysis_file=str(analysis_file),
        output_file=output_file,
        resolution=resolution,
        fps=args.fps,
        max_duration=args.max_duration,
        min_transition_interval=args.min_transition_interval,
        beat_method=args.beat_method
    )
    
    print("\n" + "=" * 70)
    print("✓ VIDEO GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ReelMaker - Create viral videos from images and audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single audio file
  python reelmaker.py analyze songs/song.mp3 --visualize
  
  # Batch analyze all songs in a directory
  python reelmaker.py batch-analyze songs/individual/
  
  # Create a video
  python reelmaker.py create images/ songs/song.mp3 -o my_video.mp4
  
  # Create a landscape video
  python reelmaker.py create images/ songs/song.mp3 --resolution 1920x1080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze audio file')
    analyze_parser.add_argument('audio', help='Path to audio file')
    analyze_parser.add_argument('--visualize', action='store_true',
                               help='Create visualization plots')
    
    # Batch analyze command
    batch_parser = subparsers.add_parser('batch-analyze', 
                                         help='Analyze multiple audio files')
    batch_parser.add_argument('directory', help='Directory containing audio files')
    batch_parser.add_argument('--output', help='Output directory for analysis results')
    
    # Create video command
    create_parser = subparsers.add_parser('create', help='Create video')
    create_parser.add_argument('images', help='Directory containing images')
    create_parser.add_argument('audio', help='Path to audio file')
    create_parser.add_argument('-o', '--output', help='Output video file',
                              default='output.mp4')
    create_parser.add_argument('-a', '--analysis', 
                              help='Path to analysis JSON file (auto-detected if not provided)')
    create_parser.add_argument('-r', '--resolution', 
                              help='Video resolution (WIDTHxHEIGHT, e.g., 1080x1920)',
                              default='1080x1920')
    create_parser.add_argument('--fps', type=int, default=30,
                              help='Frames per second (default: 30)')
    create_parser.add_argument('--max-duration', type=float,
                              help='Maximum duration in seconds (for testing, uses only first N seconds of audio)')
    create_parser.add_argument('--min-transition-interval', type=float, default=2.0,
                              help='Minimum time between image transitions in seconds (default: 2.0, prevents rapid flashing)')
    create_parser.add_argument('--beat-method', type=str, default='fullspectrum',
                              choices=['fullspectrum', 'percussive', 'lowfreq', 'demucs'],
                              help='Beat detection method to use (default: fullspectrum, demucs recommended for vocals)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    if args.command == 'analyze':
        analyze_audio_command(args)
    elif args.command == 'batch-analyze':
        batch_analyze_command(args)
    elif args.command == 'create':
        create_video_command(args)


if __name__ == '__main__':
    main()

"""
Split a large MP3 file into individual songs based on timestamps.
"""
import re
import subprocess
import os
from pathlib import Path


def parse_timestamp(timestamp_str):
    """Convert timestamp string (MM:SS or H:MM:SS) to seconds."""
    parts = timestamp_str.split(':')
    if len(parts) == 2:  # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # H:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    return 0


def sanitize_filename(filename):
    """Remove or replace invalid filename characters."""
    # Replace invalid characters with underscore
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    return filename


def parse_tracklist(list_file):
    """Parse the tracklist file and extract timestamps and song names."""
    songs = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('📋'):
                continue
            
            # Match pattern: timestamp - artist - title
            match = re.match(r'^(\d+:\d+(?::\d+)?)\s*-\s*(.+)$', line)
            if match:
                timestamp_str = match.group(1)
                song_info = match.group(2).strip()
                
                timestamp_seconds = parse_timestamp(timestamp_str)
                songs.append({
                    'timestamp': timestamp_seconds,
                    'timestamp_str': timestamp_str,
                    'info': song_info
                })
    
    return songs


def split_audio(input_file, output_dir, songs):
    """Split the audio file using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(songs)} songs to extract")
    print(f"Output directory: {output_dir}")
    
    for i, song in enumerate(songs):
        # Determine start and duration
        start_time = song['timestamp']
        
        # If there's a next song, calculate duration
        if i + 1 < len(songs):
            next_start = songs[i + 1]['timestamp']
            duration = next_start - start_time
        else:
            # Last song - let it go to the end
            duration = None
        
        # Create filename
        safe_name = sanitize_filename(song['info'])
        output_file = os.path.join(output_dir, f"{safe_name}.mp3")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', str(start_time),
        ]
        
        if duration:
            cmd.extend(['-t', str(duration)])
        
        cmd.extend([
            '-acodec', 'copy',  # Copy audio codec (faster)
            '-y',  # Overwrite output file if exists
            output_file
        ])
        
        print(f"\n[{i+1}/{len(songs)}] Extracting: {song['info']}")
        print(f"  Start: {song['timestamp_str']} ({start_time}s)")
        if duration:
            print(f"  Duration: {duration}s")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ Saved to: {output_file}")
            else:
                print(f"  ✗ Error: {result.stderr}")
        except FileNotFoundError:
            print("  ✗ Error: ffmpeg not found. Please install ffmpeg first.")
            return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return True


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    songs_dir = base_dir / 'songs'
    list_file = songs_dir / 'list.txt'
    input_mp3 = songs_dir / 'ViralSongs.mp3'
    output_dir = songs_dir / 'individual'
    
    # Verify files exist
    if not list_file.exists():
        print(f"Error: {list_file} not found")
        return
    
    if not input_mp3.exists():
        print(f"Error: {input_mp3} not found")
        return
    
    # Parse the tracklist
    print("Parsing tracklist...")
    songs = parse_tracklist(list_file)
    
    if not songs:
        print("No songs found in tracklist")
        return
    
    # Remove duplicates (the list has songs repeated)
    # Find the first duplicate to determine where the loop starts
    seen_timestamps = {}
    unique_songs = []
    
    for song in songs:
        key = f"{song['timestamp_str']}_{song['info']}"
        if key not in seen_timestamps:
            seen_timestamps[key] = True
            unique_songs.append(song)
        else:
            # Found first duplicate, stop here
            break
    
    print(f"Found {len(unique_songs)} unique songs (removed duplicates)")
    
    # Split the audio
    print("\nStarting audio split...")
    success = split_audio(str(input_mp3), str(output_dir), unique_songs)
    
    if success:
        print(f"\n✓ Successfully split {len(unique_songs)} songs!")
        print(f"  Output directory: {output_dir}")
    else:
        print("\n✗ Failed to split songs")


if __name__ == '__main__':
    main()

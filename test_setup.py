"""
Quick test to verify ReelMaker setup is working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("REELMAKER SETUP TEST")
print("=" * 70)

# Test 1: Check Python version
print("\n1. Python Version")
print(f"   {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  Warning: Python 3.8+ recommended")
else:
    print("   ✓ OK")

# Test 2: Check dependencies
print("\n2. Checking Dependencies")
dependencies = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'librosa': 'librosa',
    'matplotlib': 'matplotlib',
    'cv2': 'opencv-python',
    'PIL': 'Pillow'
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")
        missing.append(package)

if missing:
    print(f"\n   Please install: pip install {' '.join(missing)}")

# Test 3: Check ffmpeg
print("\n3. Checking ffmpeg")
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"   ✓ {version_line}")
    else:
        print("   ✗ ffmpeg not working properly")
except FileNotFoundError:
    print("   ✗ ffmpeg NOT FOUND")
    print("   Please install ffmpeg:")
    print("   - Windows: choco install ffmpeg")
    print("   - Mac: brew install ffmpeg")
    print("   - Linux: sudo apt install ffmpeg")
except Exception as e:
    print(f"   ✗ Error checking ffmpeg: {e}")

# Test 4: Check files
print("\n4. Checking Project Files")
required_files = [
    'reelmaker.py',
    'audio_analyzer.py',
    'video_generator.py',
    'split_songs.py',
    'README.md'
]

for filename in required_files:
    if Path(filename).exists():
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} - NOT FOUND")

# Test 5: Check songs directory
print("\n5. Checking Data Directories")
songs_dir = Path("songs/individual")
if songs_dir.exists():
    mp3_files = list(songs_dir.glob("*.mp3"))
    print(f"   ✓ songs/individual/ ({len(mp3_files)} MP3 files)")
else:
    print("   ⚠️  songs/individual/ not found")
    print("      Run split_songs.py to create it")

# Summary
print("\n" + "=" * 70)
print("SETUP TEST COMPLETE")
print("=" * 70)

if not missing:
    print("\n✓ All dependencies installed!")
    print("\nYou can now:")
    print("  1. Run: python reelmaker.py analyze <audio_file>")
    print("  2. Run: python reelmaker.py create <images_dir> <audio_file>")
    print("  3. Read README.md for full instructions")
else:
    print("\n⚠️  Some dependencies are missing")
    print(f"   Install with: pip install {' '.join(missing)}")

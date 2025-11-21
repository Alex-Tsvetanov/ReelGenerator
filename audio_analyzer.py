"""
Audio analysis for rhythm detection and energy peak identification.

This module analyzes audio files to detect:
- Tempo (BPM)
- Beat locations
- Small peaks (for image effects like pulsation, hue shift, rotation)
- Big peaks (for transitions between images)
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json


class AudioAnalyzer:
    """Analyze audio files for rhythm and energy peaks."""
    
    def __init__(self, audio_path: str):
        """
        Initialize the audio analyzer.
        
        Args:
            audio_path: Path to the audio file
        """
        self.audio_path = Path(audio_path)
        self.y = None  # Audio time series
        self.sr = None  # Sample rate
        self.tempo = None
        self.beat_frames = None
        self.beat_times = None
        self.onset_strength = None
        self.spectral_energy = None
        
    def load_audio(self):
        """Load the audio file."""
        print(f"Loading audio: {self.audio_path.name}")
        self.y, self.sr = librosa.load(str(self.audio_path))
        print(f"  Duration: {len(self.y)/self.sr:.2f}s")
        print(f"  Sample rate: {self.sr} Hz")
        
    def analyze_tempo_and_beats(self):
        """
        Detect tempo and beat locations in the audio.
        Uses librosa's beat tracking algorithm.
        """
        if self.y is None:
            raise ValueError("Audio not loaded. Call load_audio() first.")
        
        print("\nAnalyzing tempo and beats...")
        
        # Compute onset strength envelope (measures energy changes)
        self.onset_strength = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Detect tempo and beat frames
        tempo_array, self.beat_frames = librosa.beat.beat_track(
            onset_envelope=self.onset_strength,
            sr=self.sr
        )
        
        # Extract scalar tempo if it's an array
        if isinstance(tempo_array, np.ndarray):
            self.tempo = float(tempo_array.item()) if tempo_array.size == 1 else float(tempo_array[0])
        else:
            self.tempo = float(tempo_array)
        
        # Convert beat frames to time
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        
        print(f"  Tempo: {self.tempo:.1f} BPM")
        print(f"  Detected {len(self.beat_times)} beats")
        
        return self.tempo, self.beat_times
    
    def detect_energy_peaks(self, 
                           small_peak_percentile: float = 70,
                           big_peak_percentile: float = 90) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect small and big energy peaks in the audio.
        
        Small peaks are used for image effects (pulsation, hue shift, etc.)
        Big peaks are used for transitions between images.
        
        Args:
            small_peak_percentile: Percentile threshold for small peaks (70 = top 30%)
            big_peak_percentile: Percentile threshold for big peaks (90 = top 10%)
            
        Returns:
            Tuple of (small_peak_times, big_peak_times)
        """
        if self.onset_strength is None:
            raise ValueError("Beats not analyzed. Call analyze_tempo_and_beats() first.")
        
        print("\nDetecting energy peaks...")
        
        # Calculate spectral energy (using RMS energy)
        hop_length = 512
        self.spectral_energy = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]
        
        # Normalize energy
        energy_normalized = (self.spectral_energy - np.min(self.spectral_energy)) / \
                           (np.max(self.spectral_energy) - np.min(self.spectral_energy))
        
        # Find local maxima in the onset strength
        from scipy.signal import find_peaks
        
        # Find all local peaks with adaptive distance based on tempo
        # Faster tempo = allow closer peaks
        min_distance = max(2, self.sr // hop_length // 8)  # More sensitive to rapid beats
        
        peaks, properties = find_peaks(
            self.onset_strength,
            distance=min_distance,
            prominence=0.1  # Require minimum prominence to avoid noise
        )
        
        # Get peak heights (onset strength values at peaks)
        peak_heights = self.onset_strength[peaks]
        
        if len(peak_heights) == 0:
            print("  Warning: No peaks detected, using beat times instead")
            return self.beat_times, self.beat_times[::4]
        
        # Determine thresholds
        small_threshold = np.percentile(peak_heights, small_peak_percentile)
        big_threshold = np.percentile(peak_heights, big_peak_percentile)
        
        # Classify peaks
        small_peak_indices = peaks[peak_heights >= small_threshold]
        big_peak_indices = peaks[peak_heights >= big_threshold]
        
        # Convert to time
        small_peak_times = librosa.frames_to_time(small_peak_indices, sr=self.sr)
        big_peak_times = librosa.frames_to_time(big_peak_indices, sr=self.sr)
        
        # If we have very few peaks, supplement with beat times
        if len(small_peak_times) < len(self.beat_times) * 0.3:
            print("  Supplementing with beat times for better coverage...")
            # Add every beat as a small peak
            all_small_peaks = np.concatenate([small_peak_times, self.beat_times])
            small_peak_times = np.unique(np.sort(all_small_peaks))
        
        # Remove big peaks from small peaks (big peaks are a subset)
        small_peak_times = np.array([t for t in small_peak_times 
                                     if not any(abs(t - bt) < 0.05 for bt in big_peak_times)])
        
        print(f"  Small peaks (effects): {len(small_peak_times)}")
        print(f"  Big peaks (transitions): {len(big_peak_times)}")
        if len(big_peak_times) > 1:
            print(f"  Average time between transitions: {np.mean(np.diff(big_peak_times)):.2f}s")
        
        return small_peak_times, big_peak_times
    
    def analyze_full(self) -> Dict:
        """
        Perform complete audio analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        self.load_audio()
        tempo, beat_times = self.analyze_tempo_and_beats()
        small_peaks, big_peaks = self.detect_energy_peaks()
        
        results = {
            'audio_file': str(self.audio_path),
            'duration': len(self.y) / self.sr,
            'sample_rate': self.sr,
            'tempo_bpm': float(tempo),
            'beat_times': beat_times.tolist(),
            'small_peak_times': small_peaks.tolist(),
            'big_peak_times': big_peaks.tolist(),
            'num_beats': len(beat_times),
            'num_small_peaks': len(small_peaks),
            'num_big_peaks': len(big_peaks),
        }
        
        return results
    
    def visualize_analysis(self, output_path: str = None):
        """
        Create a visualization of the audio analysis.
        
        Args:
            output_path: Path to save the visualization (optional)
        """
        if self.y is None:
            raise ValueError("No analysis data available")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        
        # 1. Waveform
        times = np.linspace(0, len(self.y) / self.sr, len(self.y))
        axes[0].plot(times, self.y, alpha=0.5, linewidth=0.5)
        axes[0].set_title('Audio Waveform')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlim([0, len(self.y) / self.sr])
        
        # 2. Onset strength (energy changes)
        hop_length = 512
        onset_times = librosa.frames_to_time(np.arange(len(self.onset_strength)), 
                                             sr=self.sr, hop_length=hop_length)
        axes[1].plot(onset_times, self.onset_strength, linewidth=1)
        axes[1].set_title('Onset Strength (Energy Changes)')
        axes[1].set_ylabel('Strength')
        axes[1].vlines(self.beat_times, 0, np.max(self.onset_strength), 
                      color='r', alpha=0.3, linewidth=1, label='Beats')
        axes[1].legend()
        axes[1].set_xlim([0, len(self.y) / self.sr])
        
        # 3. Spectral energy with peaks
        energy_times = librosa.frames_to_time(np.arange(len(self.spectral_energy)), 
                                              sr=self.sr, hop_length=hop_length)
        axes[2].plot(energy_times, self.spectral_energy, linewidth=1, label='Energy')
        
        # Get small and big peaks
        small_peaks, big_peaks = self.detect_energy_peaks()
        axes[2].scatter(small_peaks, 
                       np.interp(small_peaks, energy_times, self.spectral_energy),
                       color='orange', s=30, marker='o', label='Small peaks (effects)', 
                       zorder=5, alpha=0.7)
        axes[2].scatter(big_peaks,
                       np.interp(big_peaks, energy_times, self.spectral_energy),
                       color='red', s=60, marker='*', label='Big peaks (transitions)',
                       zorder=6)
        
        axes[2].set_title('Spectral Energy with Detected Peaks')
        axes[2].set_ylabel('RMS Energy')
        axes[2].legend()
        axes[2].set_xlim([0, len(self.y) / self.sr])
        
        # 4. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', 
                                       ax=axes[3], cmap='viridis')
        axes[3].set_title('Spectrogram')
        axes[3].set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=axes[3], format='%+2.0f dB')
        
        # Add tempo info to the title
        fig.suptitle(f'{self.audio_path.name} - Tempo: {self.tempo:.1f} BPM', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def analyze_song(audio_path: str, visualize: bool = False, 
                save_json: bool = True) -> Dict:
    """
    Convenience function to analyze a single song.
    
    Args:
        audio_path: Path to the audio file
        visualize: Whether to create a visualization
        save_json: Whether to save results to JSON
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = AudioAnalyzer(audio_path)
    results = analyzer.analyze_full()
    
    # Save JSON
    if save_json:
        audio_path_obj = Path(audio_path)
        json_path = audio_path_obj.parent / f"{audio_path_obj.stem}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis saved to: {json_path}")
    
    # Create visualization
    if visualize:
        audio_path_obj = Path(audio_path)
        viz_path = audio_path_obj.parent / f"{audio_path_obj.stem}_analysis.png"
        analyzer.visualize_analysis(str(viz_path))
    
    return results


def batch_analyze_songs(songs_dir: str, output_dir: str = None):
    """
    Analyze multiple songs in a directory.
    
    Args:
        songs_dir: Directory containing audio files
        output_dir: Directory to save analysis results (optional)
    """
    songs_path = Path(songs_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = songs_path / 'analysis'
        output_path.mkdir(exist_ok=True)
    
    # Find all audio files
    audio_files = list(songs_path.glob('*.mp3')) + \
                 list(songs_path.glob('*.wav')) + \
                 list(songs_path.glob('*.flac'))
    
    print(f"Found {len(audio_files)} audio files to analyze\n")
    print("=" * 70)
    
    all_results = {}
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 70)
        
        try:
            analyzer = AudioAnalyzer(str(audio_file))
            results = analyzer.analyze_full()
            
            # Save individual JSON
            json_path = output_path / f"{audio_file.stem}_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create visualization
            viz_path = output_path / f"{audio_file.stem}_analysis.png"
            analyzer.visualize_analysis(str(viz_path))
            
            all_results[audio_file.name] = results
            
            print(f"✓ Completed: {audio_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {audio_file.name}: {e}")
            continue
    
    # Save summary JSON
    summary_path = output_path / 'all_songs_analysis.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Batch analysis complete!")
    print(f"  Results saved to: {output_path}")
    print(f"  Analyzed {len(all_results)} / {len(audio_files)} songs")


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific file
        audio_file = sys.argv[1]
        print(f"Analyzing: {audio_file}")
        results = analyze_song(audio_file, visualize=True, save_json=True)
        
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Duration: {results['duration']:.2f}s")
        print(f"Tempo: {results['tempo_bpm']:.1f} BPM")
        print(f"Beats: {results['num_beats']}")
        print(f"Small peaks (effects): {results['num_small_peaks']}")
        print(f"Big peaks (transitions): {results['num_big_peaks']}")
    else:
        # Batch analyze all songs
        songs_dir = Path(__file__).parent / 'songs' / 'individual'
        if songs_dir.exists():
            batch_analyze_songs(str(songs_dir))
        else:
            print("No songs directory found. Please provide an audio file path.")
            print("Usage: python audio_analyzer.py <audio_file>")

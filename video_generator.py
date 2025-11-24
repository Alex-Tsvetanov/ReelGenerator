"""
Video generator that creates viral-style videos by synchronizing images with audio peaks.

This module:
- Loads images and audio analysis data
- Applies effects (pulsation, hue shift, rotation) on small peaks
- Applies transitions (fade, slide) on big peaks
- Generates the final video
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from dataclasses import dataclass
from enum import Enum


class EffectType(Enum):
    """Types of effects for small peaks."""
    PULSE = "pulse"
    HUE_SHIFT = "hue_shift"
    ROTATE = "rotate"
    ZOOM = "zoom"
    BLUR_PULSE = "blur_pulse"


class TransitionType(Enum):
    """Types of transitions for big peaks."""
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


@dataclass
class EffectEvent:
    """An effect event at a specific time."""
    time: float
    effect_type: EffectType
    duration: float = 0.15  # Effect duration in seconds
    intensity: float = 1.0  # Effect intensity (0-1)


@dataclass
class TransitionEvent:
    """A transition event at a specific time."""
    time: float
    from_image_idx: int
    to_image_idx: int
    transition_type: TransitionType
    duration: float = 0.3  # Transition duration in seconds


class VideoGenerator:
    """Generate videos from images and audio analysis."""

    # Maximum dimension for input images to prevent memory issues
    MAX_IMAGE_DIMENSION = 4000

    def __init__(self,
                 images_dir: str,
                 audio_file: str,
                 analysis_file: str,
                 output_file: str,
                 resolution: Tuple[int, int] = (1080, 1920),  # Portrait (width, height)
                 fps: int = 30,
                 max_duration: float = None,
                 min_transition_interval: float = 2.0):
        """
        Initialize the video generator.

        Args:
            images_dir: Directory containing input images
            audio_file: Path to audio file
            analysis_file: Path to audio analysis JSON
            output_file: Path for output video
            resolution: Video resolution (width, height)
            fps: Frames per second
            max_duration: Maximum duration in seconds (for testing, uses only first N seconds)
            min_transition_interval: Minimum time (seconds) between transitions (default 2.0)
        """
        self.images_dir = Path(images_dir)
        self.audio_file = Path(audio_file)
        self.analysis_file = Path(analysis_file)
        self.output_file = Path(output_file)
        self.resolution = resolution
        self.fps = fps
        self.max_duration = max_duration
        self.min_transition_interval = min_transition_interval

        self.images = []
        self.image_backgrounds = []  # Blurred backgrounds for each image
        self.padding_masks = []  # Boolean masks marking letterbox padding areas
        self.analysis_data = None
        self.effect_events = []
        self.transition_events = []

    def sanitize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Sanitize a frame to ensure it meets exact requirements.
        Fixes corruption from PIL/NumPy/OpenCV dimension mismatches.

        Args:
            frame: Input frame (may have wrong shape, dtype, or memory layout)

        Returns:
            Sanitized frame with exact target resolution, uint8 dtype, contiguous memory
        """
        target_w, target_h = self.resolution

        # Ensure frame is a numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Ensure 3 channels (RGB)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        current_h, current_w = frame.shape[:2]

        # If dimensions don't match exactly, resize
        if current_h != target_h or current_w != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Ensure contiguous memory layout (fixes OpenCV blend artifacts)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Final validation
        assert frame.shape == (target_h, target_w, 3), f"Frame shape mismatch: {frame.shape} != {(target_h, target_w, 3)}"
        assert frame.dtype == np.uint8, f"Frame dtype mismatch: {frame.dtype} != uint8"

        return frame

    def load_images(self):
        """Load all images from the images directory."""
        print(f"Loading images from: {self.images_dir}")

        # Increase PIL decompression bomb limit for large images
        Image.MAX_IMAGE_PIXELS = None  # Remove limit (or set to a high value like 500000000)

        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(ext))
            image_files.extend(self.images_dir.glob(ext.upper()))

        # Sort images
        image_files = sorted(image_files)

        print(f"Found {len(image_files)} images")

        # Load and resize images
        skipped = 0
        for i, img_path in enumerate(image_files, 1):
            try:
                # Load image
                img = Image.open(img_path)

                # Apply EXIF orientation if present (fixes rotated phone photos)
                try:
                    img = ImageOps.exif_transpose(img)
                except:
                    pass  # If EXIF handling fails, continue with original

                # Convert to RGB
                img = img.convert('RGB')

                # Downscale very large images to prevent memory issues
                img_w, img_h = img.size
                max_dim = max(img_w, img_h)
                if max_dim > self.MAX_IMAGE_DIMENSION:
                    scale = self.MAX_IMAGE_DIMENSION / max_dim
                    new_w = int(img_w * scale)
                    new_h = int(img_h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    print(f"  [{i}/{len(image_files)}] Downscaled {img_path.name} from {max_dim}px to {max(new_w, new_h)}px")
                elif img.size[0] * img.size[1] > 50000000:  # 50 megapixels
                    print(f"  [{i}/{len(image_files)}] Large image detected: {img_path.name} ({img.size[0]}x{img.size[1]}) - resizing...")

                # Create foreground, background, and padding mask
                foreground, background, padding_mask = self._create_image_with_background(img)

                # Convert to numpy arrays and sanitize for consistent format
                fg_array = self.sanitize_frame(np.array(foreground))
                bg_array = self.sanitize_frame(np.array(background))
                self.images.append(fg_array)
                self.image_backgrounds.append(bg_array)
                self.padding_masks.append(padding_mask)

            except Exception as e:
                print(f"  ✗ Error loading {img_path.name}: {e}")
                skipped += 1
                continue

        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} images due to errors")

        print(f"Loaded {len(self.images)} images")

        if len(self.images) == 0:
            raise ValueError("No images found in directory")

    def _create_image_with_background(self, img: Image.Image) -> tuple:
        """Create foreground image, blurred background, and padding mask."""
        target_w, target_h = self.resolution
        img_w, img_h = img.size

        # Create blurred background - fit to fill entire frame
        bg_scale = max(target_w / img_w, target_h / img_h)
        bg_w = int(img_w * bg_scale)
        bg_h = int(img_h * bg_scale)
        background = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

        # Crop background to exact size if needed
        if bg_w > target_w or bg_h > target_h:
            left = (bg_w - target_w) // 2
            top = (bg_h - target_h) // 2
            background = background.crop((left, top, left + target_w, top + target_h))

        # Apply blur to background
        background = background.filter(ImageFilter.GaussianBlur(radius=20))

        # Create foreground - fit to width
        fg_scale = target_w / img_w
        fg_w = target_w
        fg_h = int(img_h * fg_scale)
        foreground = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

        # Create padding mask - True where there's letterbox padding
        padding_mask = np.zeros((target_h, target_w), dtype=bool)

        # If foreground is taller than target, crop from center
        if fg_h > target_h:
            crop_y = (fg_h - target_h) // 2
            foreground = foreground.crop((0, crop_y, fg_w, crop_y + target_h))
            # No padding in this case
        # If foreground is shorter, we'll composite it on background later
        elif fg_h < target_h:
            # Create a new foreground with black bars
            final_fg = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            paste_y = (target_h - fg_h) // 2
            final_fg.paste(foreground, (0, paste_y))
            foreground = final_fg

            # Mark the black bars (top and bottom) as padding
            padding_mask[:paste_y, :] = True  # Top padding
            padding_mask[paste_y + fg_h:, :] = True  # Bottom padding

        return foreground, background, padding_mask

    def load_analysis(self):
        """Load audio analysis data."""
        print(f"Loading analysis from: {self.analysis_file}")

        with open(self.analysis_file, 'r') as f:
            self.analysis_data = json.load(f)

        print(f"  Duration: {self.analysis_data['duration']:.2f}s")
        print(f"  Tempo: {self.analysis_data['tempo_bpm']:.1f} BPM")
        print(f"  Small peaks: {self.analysis_data['num_small_peaks']}")
        print(f"  Big peaks: {self.analysis_data['num_big_peaks']}")

    def plan_effects_and_transitions(self, min_transition_interval: float = 2.0):
        """
        Plan when effects and transitions should occur.
        
        Args:
            min_transition_interval: Minimum time (seconds) between transitions to prevent rapid flashing
        """
        if not self.analysis_data:
            raise ValueError("Analysis data not loaded")

        if not self.images:
            raise ValueError("Images not loaded")

        print("\nPlanning effects and transitions...")

        # Get peak times
        small_peaks = self.analysis_data['small_peak_times']
        big_peaks = self.analysis_data['big_peak_times']

        # Filter big peaks to enforce minimum interval between transitions
        filtered_big_peaks = []
        last_peak_time = -min_transition_interval  # Allow first peak
        
        for peak_time in big_peaks:
            if peak_time - last_peak_time >= min_transition_interval:
                filtered_big_peaks.append(peak_time)
                last_peak_time = peak_time
        
        if len(filtered_big_peaks) < len(big_peaks):
            print(f"  Filtered {len(big_peaks) - len(filtered_big_peaks)} transitions (too close together)")
        
        big_peaks = filtered_big_peaks

        # Create effect events for small peaks
        effect_types = list(EffectType)
        for i, peak_time in enumerate(small_peaks):
            effect_type = effect_types[i % len(effect_types)]
            self.effect_events.append(
                EffectEvent(
                    time=peak_time,
                    effect_type=effect_type,
                    duration=0.15,
                    intensity=0.8 + np.random.random() * 0.2  # Random intensity
                )
            )

        # Create transition events for big peaks
        transition_types = list(TransitionType)
        num_images = len(self.images)

        for i, peak_time in enumerate(big_peaks):
            from_idx = i % num_images
            to_idx = (i + 1) % num_images
            transition_type = transition_types[i % len(transition_types)]

            self.transition_events.append(
                TransitionEvent(
                    time=peak_time,
                    from_image_idx=from_idx,
                    to_image_idx=to_idx,
                    transition_type=transition_type,
                    duration=0.3
                )
            )

        print(f"  Planned {len(self.effect_events)} effects")
        print(f"  Planned {len(self.transition_events)} transitions")
        if self.transition_events:
            avg_interval = self.analysis_data['duration'] / len(self.transition_events)
            print(f"  Average time per image: {avg_interval:.2f}s")

    def apply_effect(self, frame: np.ndarray, padding_mask: np.ndarray,
                    effect: EffectEvent, progress: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply an effect to a frame and transform the padding mask accordingly.

        Args:
            frame: Input frame (numpy array, RGB)
            padding_mask: Boolean mask marking padding areas
            effect: Effect to apply
            progress: Progress through effect (0-1)

        Returns:
            Tuple of (modified frame with alpha channel (RGBA), transformed padding mask)
        """
        # Convert to PIL with alpha channel for transparency tracking
        img = Image.fromarray(frame).convert('RGBA')

        # Convert padding mask to PIL image (255 = padding, 0 = content)
        mask_img = Image.fromarray((padding_mask * 255).astype(np.uint8), mode='L')

        # Calculate effect strength (ease in-out)
        strength = np.sin(progress * np.pi) * effect.intensity

        if effect.effect_type == EffectType.PULSE:
            # Scale image with edge handling to avoid black glitches
            scale = 1.0 + strength * 0.15
            w, h = img.size
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Only scale if we're zooming in (scale > 1)
            if scale > 1.0:
                img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                mask_scaled = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
                # Center crop back to original size
                left = max(0, (new_w - w) // 2)
                top = max(0, (new_h - h) // 2)
                img = img_scaled.crop((left, top, left + w, top + h))
                mask_img = mask_scaled.crop((left, top, left + w, top + h))
            # For scale < 1, keep original to avoid black borders

        elif effect.effect_type == EffectType.HUE_SHIFT:
            # Adjust color saturation
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 + strength * 0.5)

        elif effect.effect_type == EffectType.ROTATE:
            # Rotate image with transparent fill for corners
            angle = strength * 5.0  # Max 5 degrees
            # Rotate both image and mask with the same parameters
            img = img.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
            # Rotate mask (255 = padding after rotation becomes padding)
            mask_img = mask_img.rotate(angle, expand=False, resample=Image.Resampling.NEAREST, fillcolor=255)

        elif effect.effect_type == EffectType.ZOOM:
            # Zoom in effect with edge handling
            scale = 1.0 + strength * 0.2
            w, h = img.size
            new_w = int(w * scale)
            new_h = int(h * scale)

            if scale > 1.0:
                img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                mask_scaled = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
                # Center crop
                left = max(0, (new_w - w) // 2)
                top = max(0, (new_h - h) // 2)
                img = img_scaled.crop((left, top, left + w, top + h))
                mask_img = mask_scaled.crop((left, top, left + w, top + h))

        elif effect.effect_type == EffectType.BLUR_PULSE:
            # Apply motion blur
            blur_amount = strength * 3.0
            if blur_amount > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))

        # Convert to numpy arrays
        result = np.array(img)  # RGBA with alpha channel
        transformed_mask = np.array(mask_img) > 127  # Convert back to boolean mask
        # Note: RGBA result will be composited with background in main loop
        return result, transformed_mask

    def apply_transition(self, frame1: np.ndarray, frame2: np.ndarray,
                        bg1: np.ndarray, bg2: np.ndarray,
                        mask1: np.ndarray, mask2: np.ndarray,
                        transition: TransitionEvent, progress: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a transition between two frames with their backgrounds and padding masks.

        Args:
            frame1: First foreground frame
            frame2: Second foreground frame
            bg1: First background frame
            bg2: Second background frame
            mask1: First padding mask
            mask2: Second padding mask
            transition: Transition to apply
            progress: Progress through transition (0-1)

        Returns:
            Tuple of (blended foreground, blended background)
        """
        h, w = frame1.shape[:2]

        if transition.transition_type == TransitionType.FADE:
            # Composite each frame with its background first to avoid black padding
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]
            
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]
            
            # Now crossfade the composited images
            alpha = progress
            blended = cv2.addWeighted(composite1, 1 - alpha, composite2, alpha, 0)
            blended_bg = cv2.addWeighted(bg1, 1 - alpha, bg2, alpha, 0)
            return blended, blended_bg

        elif transition.transition_type == TransitionType.SLIDE_LEFT:
            # Slide from right to left - composite each frame with its background first
            offset = int(w * progress)

            # Composite frame1 with bg1
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Now slide the composited images
            result = np.zeros_like(frame1)
            result_bg = np.zeros_like(bg1)
            if offset < w and offset > 0:
                w1_slice = min(w - offset, composite1.shape[1] - offset)
                w2_slice = min(offset, composite2.shape[1])
                result[:, :w1_slice] = composite1[:, offset:offset + w1_slice]
                result[:, w-w2_slice:] = composite2[:, :w2_slice]
                result_bg[:, :w1_slice] = bg1[:, offset:offset + w1_slice]
                result_bg[:, w-w2_slice:] = bg2[:, :w2_slice]
            else:
                result = composite2.copy()
                result_bg = bg2.copy()

            # Return composited result and background
            return result, result_bg

        elif transition.transition_type == TransitionType.SLIDE_RIGHT:
            # Slide from left to right - composite each frame with its background first
            offset = int(w * progress)

            # Composite frame1 with bg1
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Now slide the composited images
            result = np.zeros_like(frame1)
            result_bg = np.zeros_like(bg1)
            if offset < w and offset > 0:
                w1_slice = min(w - offset, composite1.shape[1])
                w2_slice = min(offset, composite2.shape[1] - (w - offset))
                result[:, offset:offset + w1_slice] = composite1[:, :w1_slice]
                result[:, :w2_slice] = composite2[:, w-w2_slice:w]
                result_bg[:, offset:offset + w1_slice] = bg1[:, :w1_slice]
                result_bg[:, :w2_slice] = bg2[:, w-w2_slice:w]
            else:
                result = composite2.copy()
                result_bg = bg2.copy()

            return result, result_bg

        elif transition.transition_type == TransitionType.SLIDE_UP:
            # Slide from bottom to top - composite each frame with its background first
            offset = int(h * progress)

            # Composite frame1 with bg1
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Now slide the composited images
            result = np.zeros_like(frame1)
            result_bg = np.zeros_like(bg1)
            if offset < h and offset > 0:
                h1_slice = min(h - offset, composite1.shape[0] - offset)
                h2_slice = min(offset, composite2.shape[0])
                result[:h1_slice, :] = composite1[offset:offset + h1_slice, :]
                result[h-h2_slice:, :] = composite2[:h2_slice, :]
                result_bg[:h1_slice, :] = bg1[offset:offset + h1_slice, :]
                result_bg[h-h2_slice:, :] = bg2[:h2_slice, :]
            else:
                result = composite2.copy()
                result_bg = bg2.copy()

            return result, result_bg

        elif transition.transition_type == TransitionType.SLIDE_DOWN:
            # Slide from top to bottom - composite each frame with its background first
            offset = int(h * progress)

            # Composite frame1 with bg1
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Now slide the composited images
            result = np.zeros_like(frame1)
            result_bg = np.zeros_like(bg1)
            if offset < h and offset > 0:
                h1_slice = min(h - offset, composite1.shape[0])
                h2_slice = min(offset, composite2.shape[0] - (h - offset))
                result[offset:offset + h1_slice, :] = composite1[:h1_slice, :]
                result[:h2_slice, :] = composite2[h-h2_slice:h, :]
                result_bg[offset:offset + h1_slice, :] = bg1[:h1_slice, :]
                result_bg[:h2_slice, :] = bg2[h-h2_slice:h, :]
            else:
                result = composite2.copy()
                result_bg = bg2.copy()

            return result, result_bg

        elif transition.transition_type == TransitionType.ZOOM_IN:
            # Zoom in from frame1 to frame2 - composite first, then zoom
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            scale = 1.0 + progress * 0.5
            img1 = Image.fromarray(composite1)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_scaled = img1.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center crop
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img_cropped = img_scaled.crop((left, top, left + w, top + h))

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Fade to frame2
            blended = cv2.addWeighted(np.array(img_cropped), 1 - progress,
                                     composite2, progress, 0)
            blended_bg = cv2.addWeighted(bg1, 1 - progress, bg2, progress, 0)
            return blended, blended_bg

        elif transition.transition_type == TransitionType.ZOOM_OUT:
            # Zoom out from frame1 to frame2 - composite first, then zoom
            composite1 = bg1.copy()
            composite1[~mask1] = frame1[~mask1]

            scale = 1.5 - progress * 0.5
            img1 = Image.fromarray(composite1)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > w and new_h > h:
                img_scaled = img1.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img_cropped = img_scaled.crop((left, top, left + w, top + h))
            else:
                img_cropped = img1

            # Composite frame2 with bg2
            composite2 = bg2.copy()
            composite2[~mask2] = frame2[~mask2]

            # Fade to frame2
            blended = cv2.addWeighted(np.array(img_cropped), 1 - progress,
                                     composite2, progress, 0)
            blended_bg = cv2.addWeighted(bg1, 1 - progress, bg2, progress, 0)
            return blended, blended_bg

        return frame2, bg2

    def generate_video(self):
        """Generate the final video."""
        print("\nGenerating video...")

        duration = self.analysis_data['duration']
        
        # Apply max_duration limit if specified
        if self.max_duration is not None and self.max_duration < duration:
            duration = self.max_duration
            print(f"  Limiting duration to {duration:.2f}s for testing")
        
        total_frames = int(duration * self.fps)

        # Setup video writer - use lossless codec for temp to avoid artifacts
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless codec
        temp_video = str(self.output_file.with_suffix('.temp.avi'))
        out = cv2.VideoWriter(temp_video, fourcc, self.fps, self.resolution)

        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Resolution: {self.resolution[0]}x{self.resolution[1]}")
        print(f"  FPS: {self.fps}")

        # Track current image and active effects
        current_image_idx = 0

        for frame_num in range(total_frames):
            current_time = frame_num / self.fps

            # Update current image based on completed transitions
            for trans in self.transition_events:
                if current_time >= trans.time + trans.duration:
                    # Transition completed, use the destination image
                    current_image_idx = trans.to_image_idx

            # Step 1: Fetch base foreground (RGB) and background (RGB)
            base_foreground_rgb = self.images[current_image_idx].copy()
            background_rgb = self.image_backgrounds[current_image_idx].copy()

            # Step 2: Check for active transitions
            active_transition = None
            for trans in self.transition_events:
                if trans.time <= current_time < trans.time + trans.duration:
                    active_transition = trans
                    progress = (current_time - trans.time) / trans.duration

                    # Apply transition with foreground, background, and padding masks
                    frame1 = self.images[trans.from_image_idx]
                    frame2 = self.images[trans.to_image_idx]
                    bg1 = self.image_backgrounds[trans.from_image_idx]
                    bg2 = self.image_backgrounds[trans.to_image_idx]
                    mask1 = self.padding_masks[trans.from_image_idx]
                    mask2 = self.padding_masks[trans.to_image_idx]

                    # Transition returns already composited result for slides
                    result_fg, result_bg = self.apply_transition(
                        frame1, frame2, bg1, bg2, mask1, mask2, trans, progress
                    )

                    # For slide transitions, result is already composited, use directly
                    if trans.transition_type in [TransitionType.SLIDE_LEFT, TransitionType.SLIDE_RIGHT,
                                                  TransitionType.SLIDE_UP, TransitionType.SLIDE_DOWN,
                                                  TransitionType.ZOOM_IN, TransitionType.ZOOM_OUT]:
                        # Result is already composited, so we'll handle it specially
                        frame = result_fg
                        background_rgb = result_bg
                        # Skip step 4 by using a flag
                        base_foreground_rgb = None
                    else:
                        # For FADE, we got blended foreground/background separately
                        base_foreground_rgb = result_fg
                        background_rgb = result_bg
                    break

            # Step 3: Apply effects to foreground (returns RGBA with alpha channel and transformed mask)
            foreground_rgba = None
            transformed_padding_mask = self.padding_masks[current_image_idx]
            has_effect = False

            if active_transition is None:  # Only apply effects when not transitioning
                for effect in self.effect_events:
                    if effect.time <= current_time < effect.time + effect.duration:
                        progress = (current_time - effect.time) / effect.duration
                        # apply_effect converts RGB to RGBA and returns transformed image and mask
                        foreground_rgba, transformed_padding_mask = self.apply_effect(
                            base_foreground_rgb,
                            self.padding_masks[current_image_idx],
                            effect,
                            progress
                        )
                        has_effect = True
                        break  # Only one effect at a time

            # Step 4: Composite foreground over background
            # Skip if already composited by transition (slide/zoom)
            if base_foreground_rgb is None:
                # Frame already set by transition, skip compositing
                pass
            elif has_effect and foreground_rgba is not None:
                # Start with background as the base
                frame = background_rgb.copy()
                # Foreground has been transformed and is RGBA
                # Extract RGB and alpha channels
                foreground_rgb = foreground_rgba[:, :, :3]
                alpha_channel = foreground_rgba[:, :, 3]

                # Use the transformed padding mask (accounts for rotation, scaling, etc.)
                # Create mask: where alpha > 0 (non-transparent pixels) AND not in padding
                opaque_mask = (alpha_channel > 0) & (~transformed_padding_mask)

                # Place foreground pixels over background only where opaque and not padding
                # This preserves the blurred background in letterbox areas and rotated corners
                frame[opaque_mask] = foreground_rgb[opaque_mask]
            else:
                # Start with background as the base
                frame = background_rgb.copy()
                # No effects applied, use original foreground (RGB)
                # Use the stored padding mask (marks letterbox areas)
                padding_mask = self.padding_masks[current_image_idx]

                # Place foreground over background, skipping padding areas
                frame[~padding_mask] = base_foreground_rgb[~padding_mask]

            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

            # Progress indicator
            if frame_num % (self.fps * 5) == 0:  # Every 5 seconds
                progress_pct = (frame_num / total_frames) * 100
                print(f"  Progress: {progress_pct:.1f}% ({current_time:.1f}s / {duration:.1f}s)")

        out.release()

        print(f"\n✓ Video frames generated: {temp_video}")

        # Add audio using ffmpeg
        print("Adding audio to video...")
        final_output = str(self.output_file)

        import subprocess
        cmd = [
            'ffmpeg', '-i', temp_video, '-i', str(self.audio_file),
            '-c:v', 'libx264',
            '-preset', 'slow',  # Better quality encoding
            '-crf', '18',  # High quality (0-51, lower = better)
            '-pix_fmt', 'yuv420p',  # Compatibility with most players
            '-c:a', 'aac', '-b:a', '192k',  # High quality audio
            '-shortest', '-y', final_output
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Final video saved: {final_output}")

            # Remove temp file
            Path(temp_video).unlink()
        else:
            print(f"✗ Error adding audio: {result.stderr}")
            print(f"  Video without audio saved as: {temp_video}")

    def create_video(self):
        """Complete workflow to create a video."""
        self.load_images()
        self.load_analysis()
        self.plan_effects_and_transitions(min_transition_interval=self.min_transition_interval)
        self.generate_video()


def create_viral_video(images_dir: str,
                      audio_file: str,
                      analysis_file: str = None,
                      output_file: str = "output.mp4",
                      resolution: Tuple[int, int] = (1080, 1920),
                      fps: int = 30,
                      max_duration: float = None,
                      min_transition_interval: float = 2.0):
    """
    Convenience function to create a viral video.

    Args:
        images_dir: Directory containing images
        audio_file: Path to audio file
        analysis_file: Path to analysis JSON (if None, will look for matching file)
        output_file: Output video path
        resolution: Video resolution (width, height)
        fps: Frames per second
        max_duration: Maximum duration in seconds (for testing, uses only first N seconds)
        min_transition_interval: Minimum time (seconds) between transitions (default 2.0)
    """
    # Find analysis file if not provided
    if analysis_file is None:
        audio_path = Path(audio_file)
        analysis_file = audio_path.parent / f"{audio_path.stem}_analysis.json"

        if not analysis_file.exists():
            raise ValueError(f"Analysis file not found: {analysis_file}")

    generator = VideoGenerator(
        images_dir=images_dir,
        audio_file=audio_file,
        analysis_file=analysis_file,
        output_file=output_file,
        resolution=resolution,
        fps=fps,
        max_duration=max_duration,
        min_transition_interval=min_transition_interval
    )

    generator.create_video()

    return output_file


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python video_generator.py <images_dir> <audio_file> [output_file]")
        print("\nExample:")
        print('  python video_generator.py images/ "songs/individual/song.mp3" output.mp4')
        sys.exit(1)

    images_dir = sys.argv[1]
    audio_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"

    print("=" * 70)
    print("VIRAL VIDEO GENERATOR")
    print("=" * 70)

    create_viral_video(
        images_dir=images_dir,
        audio_file=audio_file,
        output_file=output_file,
        resolution=(1080, 1920),  # Portrait for TikTok/Reels
        fps=30
    )

    print("\n" + "=" * 70)
    print("✓ VIDEO GENERATION COMPLETE!")
    print("=" * 70)

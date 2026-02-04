"""Video clip recorder for fall events."""

import time
import logging
import threading
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    """Frame stored in ring buffer."""
    data: np.ndarray
    timestamp: float


class ClipRecorder:
    """
    Records video clips around fall detection events.

    Uses a ring buffer to store recent frames, then saves
    pre-event and post-event footage when triggered.
    """

    def __init__(
        self,
        output_dir: Path,
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
        fps: float = 30.0
    ):
        """
        Initialize clip recorder.

        Args:
            output_dir: Directory to save clips
            pre_seconds: Seconds of footage before event
            post_seconds: Seconds of footage after event
            fps: Expected frame rate
        """
        self.output_dir = Path(output_dir)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps

        # Ring buffer for pre-event frames
        buffer_size = int(pre_seconds * fps) + 10
        self._buffer: Deque[BufferedFrame] = deque(maxlen=buffer_size)

        # Recording state
        self._recording = False
        self._post_frames: list[BufferedFrame] = []
        self._post_frame_count = 0
        self._post_frame_target = int(post_seconds * fps)
        self._trigger_timestamp: Optional[float] = None
        self._event_name: str = ""

        # Thread safety
        self._lock = threading.Lock()

        # Enabled flag
        self._enabled = False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def enable(self) -> None:
        """Enable clip recording."""
        self._enabled = True
        logger.info(f"Clip recorder enabled. Output: {self.output_dir}")

    def disable(self) -> None:
        """Disable clip recording."""
        self._enabled = False

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Add a frame to the buffer.

        Args:
            frame: BGR frame from OpenCV
            timestamp: Frame timestamp
        """
        if not self._enabled:
            return

        buffered = BufferedFrame(data=frame.copy(), timestamp=timestamp)

        with self._lock:
            if self._recording:
                # Collecting post-event frames
                self._post_frames.append(buffered)
                self._post_frame_count += 1

                if self._post_frame_count >= self._post_frame_target:
                    # Done collecting, save clip
                    self._save_clip()
            else:
                # Normal buffering
                self._buffer.append(buffered)

    def trigger(self, event_name: str = "fall") -> bool:
        """
        Trigger clip recording.

        Args:
            event_name: Name for the event (used in filename)

        Returns:
            True if recording started, False if already recording
        """
        if not self._enabled:
            return False

        with self._lock:
            if self._recording:
                logger.warning("Already recording a clip")
                return False

            self._recording = True
            self._trigger_timestamp = time.time()
            self._event_name = event_name
            self._post_frames = []
            self._post_frame_count = 0

            logger.info(f"Clip recording triggered: {event_name}")
            return True

    def _save_clip(self) -> None:
        """Save the recorded clip to file."""
        if not self._buffer and not self._post_frames:
            logger.warning("No frames to save")
            self._reset_recording()
            return

        # Generate filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self._event_name}_{timestamp_str}.mp4"
        filepath = self.output_dir / filename

        # Combine pre and post frames
        all_frames = list(self._buffer) + self._post_frames

        if not all_frames:
            logger.warning("No frames to save")
            self._reset_recording()
            return

        # Get frame dimensions from first frame
        height, width = all_frames[0].data.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))

        try:
            for buffered in all_frames:
                writer.write(buffered.data)

            logger.info(f"Saved clip: {filepath} ({len(all_frames)} frames)")
        except Exception as e:
            logger.error(f"Error saving clip: {e}")
        finally:
            writer.release()
            self._reset_recording()

    def _reset_recording(self) -> None:
        """Reset recording state."""
        self._recording = False
        self._post_frames = []
        self._post_frame_count = 0
        self._trigger_timestamp = None
        self._event_name = ""

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def is_enabled(self) -> bool:
        """Check if recorder is enabled."""
        return self._enabled

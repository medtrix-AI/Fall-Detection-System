"""Video source abstraction for webcam and video files."""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Generator

import cv2
import numpy as np

from ..config.settings import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Frame with metadata."""
    data: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int
    source_fps: float


class VideoSource:
    """
    Unified video source for webcam and video files.
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize video source.

        Args:
            config: Video configuration. Uses defaults if not provided.
        """
        self.config = config or VideoConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._source_fps: float = 30.0

    def open(self) -> bool:
        """Open the video source."""
        if self.config.source == "video":
            return self._open_video_file()
        else:
            return self._open_webcam()

    def _open_webcam(self) -> bool:
        """Open webcam source."""
        logger.info(f"Opening webcam {self.config.camera_index}...")

        self._cap = cv2.VideoCapture(self.config.camera_index)

        if not self._cap.isOpened():
            logger.error(f"Could not open webcam {self.config.camera_index}")
            return False

        # Configure resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Get actual values
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._start_time = time.time()

        logger.info(
            f"Opened webcam {self.config.camera_index} @ "
            f"{actual_width}x{actual_height} @ {self._source_fps:.1f} FPS"
        )
        return True

    def _open_video_file(self) -> bool:
        """Open video file source."""
        if not self.config.video_path:
            logger.error("No video path specified")
            return False

        path = Path(self.config.video_path)
        if not path.exists():
            logger.error(f"Video file not found: {path}")
            return False

        logger.info(f"Opening video file: {path}")

        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            logger.error(f"Could not open video: {path}")
            return False

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._start_time = time.time()

        logger.info(
            f"Opened video: {path.name} @ {width}x{height} @ "
            f"{self._source_fps:.1f} FPS ({total_frames} frames)"
        )
        return True

    def read(self) -> Tuple[bool, Optional[Frame]]:
        """
        Read next frame.

        Returns:
            Tuple of (success, Frame or None)
        """
        if not self._cap:
            return False, None

        ret, data = self._cap.read()

        if not ret or data is None:
            return False, None

        self._frame_count += 1

        frame = Frame(
            data=data,
            timestamp=time.time(),
            frame_id=self._frame_count,
            width=data.shape[1],
            height=data.shape[0],
            source_fps=self._source_fps
        )

        return True, frame

    def frames(self) -> Generator[Frame, None, None]:
        """Generator yielding frames."""
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    def release(self) -> None:
        """Release video source."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("Video source released")

    @property
    def frame_count(self) -> int:
        """Get total frames read."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """Check if video source is open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get source FPS."""
        return self._source_fps

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.release()

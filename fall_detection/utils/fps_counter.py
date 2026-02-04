"""FPS counter utility."""

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """
    Measures frames per second with smoothing.

    Uses a rolling window to calculate average FPS.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._last_time: float = time.perf_counter()
        self._fps: float = 0.0

    def tick(self) -> float:
        """
        Record a frame and return current FPS.

        Returns:
            Current FPS value
        """
        current_time = time.perf_counter()
        self._timestamps.append(current_time)

        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

        self._last_time = current_time
        return self._fps

    def reset(self) -> None:
        """Reset the FPS counter."""
        self._timestamps.clear()
        self._last_time = time.perf_counter()
        self._fps = 0.0

    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._fps

    @property
    def frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if self._fps > 0:
            return 1000.0 / self._fps
        return 0.0

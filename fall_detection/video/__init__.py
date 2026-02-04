"""Video module."""

from .source import VideoSource, Frame
from .recorder import ClipRecorder

__all__ = [
    "VideoSource",
    "Frame",
    "ClipRecorder",
]

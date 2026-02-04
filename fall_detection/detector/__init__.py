"""Detector module."""

from .base import BaseDetector, DetectionResult, BoundingBox
from .yolo_detector import YOLOFallDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "BoundingBox",
    "YOLOFallDetector",
]

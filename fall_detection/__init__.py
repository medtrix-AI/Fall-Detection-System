"""
Fall Detection System V2
YOLOv11-based fall detection using melihuzunoglu/human-fall-detection
"""

__version__ = "2.0.0"
__author__ = "Medtrix"

from .core.app import FallDetectionApp
from .detector.yolo_detector import YOLOFallDetector
from .core.state_machine import FallStateMachine, FallState

__all__ = [
    "FallDetectionApp",
    "YOLOFallDetector",
    "FallStateMachine",
    "FallState",
]

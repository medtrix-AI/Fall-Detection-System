"""Base detector interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from enum import IntEnum
import numpy as np


class DetectionClass(IntEnum):
    """Detection classes for fall detection model."""
    FALLEN = 0
    SITTING = 1
    STANDING = 2


@dataclass
class BoundingBox:
    """Bounding box with confidence and class info."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y2 - self.y1

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box as integer coordinates."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class DetectionResult:
    """Result from a single inference."""
    timestamp: float
    frame_id: int
    detections: List[BoundingBox]
    inference_time_ms: float
    frame_shape: Tuple[int, int]  # (height, width)

    @property
    def has_fallen(self) -> bool:
        """Check if any fallen detection exists."""
        return any(d.class_name == "Fallen" for d in self.detections)

    @property
    def fallen_detections(self) -> List[BoundingBox]:
        """Get all fallen detections."""
        return [d for d in self.detections if d.class_name == "Fallen"]

    @property
    def sitting_detections(self) -> List[BoundingBox]:
        """Get all sitting detections."""
        return [d for d in self.detections if d.class_name == "Sitting"]

    @property
    def standing_detections(self) -> List[BoundingBox]:
        """Get all standing detections."""
        return [d for d in self.detections if d.class_name == "Standing"]

    def get_by_class(self, class_name: str) -> List[BoundingBox]:
        """Get detections by class name."""
        return [d for d in self.detections if d.class_name == class_name]

    def get_best_fallen(self) -> BoundingBox | None:
        """Get the highest confidence fallen detection."""
        fallen = self.fallen_detections
        if not fallen:
            return None
        return max(fallen, key=lambda d: d.confidence)


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a frame."""
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Get list of class names."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

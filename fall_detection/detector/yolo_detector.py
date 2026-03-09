"""YOLOv11 Fall Detector using Hugging Face model."""

import time
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult, BoundingBox
from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)


class YOLOFallDetector(BaseDetector):
    """
    YOLOv11-based fall detector using melihuzunoglu/human-fall-detection.

    Classes: "Fallen", "Sitting", "Standing"
    """

    CLASS_NAMES = ["Fallen", "Sitting", "Standing"]

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the YOLO fall detector.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._model: Optional[YOLO] = None
        self._model_path: Optional[Path] = None
        self._frame_count = 0
        self._device: str = "cpu"

    def load_model(self) -> None:
        """Download and load the YOLO model from Hugging Face."""
        logger.info(f"Downloading model from {self.config.repo_id}...")

        # Download model from Hugging Face Hub
        self._model_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename
        )

        logger.info(f"Model downloaded to: {self._model_path}")

        # Load with Ultralytics
        self._model = YOLO(self._model_path)

        # Resolve device
        self._device = self._resolve_device(self.config.device)
        logger.info(f"Using device: {self._device}")

        logger.info("Model loaded successfully")

    def _resolve_device(self, device_str: str) -> str:
        """Resolve device string to actual device."""
        if device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA available, using GPU")
                    return "cuda"
                else:
                    if torch.backends.mps.is_available():
                        logger.info(
                            "MPS available but skipped due to known YOLO "
                            "bounding box bugs. Use --device mps to force it."
                        )
                    logger.info("Using CPU")
                    return "cpu"
            except ImportError:
                return "cpu"
        return device_str

    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Warming up model...")
        dummy_frame = np.zeros(
            (self.config.input_size, self.config.input_size, 3),
            dtype=np.uint8
        )
        _ = self._model(
            dummy_frame,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self._device
        )
        logger.info("Model warmup complete")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run fall detection on a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            DetectionResult with all detections
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._frame_count += 1
        timestamp = time.time()
        start_time = time.perf_counter()

        # Run inference
        results = self._model(
            frame,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            device=self._device
        )

        inference_time = (time.perf_counter() - start_time) * 1000

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Ensure class_id is valid
                    if cls_id < len(self.CLASS_NAMES):
                        class_name = self.CLASS_NAMES[cls_id]
                    else:
                        class_name = f"Unknown_{cls_id}"

                    detection = BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=class_name
                    )
                    detections.append(detection)

        return DetectionResult(
            timestamp=timestamp,
            frame_id=self._frame_count,
            detections=detections,
            inference_time_ms=inference_time,
            frame_shape=(frame.shape[0], frame.shape[1])
        )

    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return self.CLASS_NAMES

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def frame_count(self) -> int:
        """Get total frames processed."""
        return self._frame_count

    def reset_frame_count(self) -> None:
        """Reset frame counter."""
        self._frame_count = 0

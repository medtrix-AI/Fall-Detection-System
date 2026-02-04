"""Event logger for fall detection events in JSONL format."""

import json
import time
import logging
from pathlib import Path
from typing import Optional, TextIO, Dict, Any
from datetime import datetime

from ..core.state_machine import StateEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Logs fall detection events to JSONL (JSON Lines) format.

    Each line is a valid JSON object containing event details.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize event logger.

        Args:
            output_dir: Directory to store log files
        """
        self.output_dir = Path(output_dir)
        self._file: Optional[TextIO] = None
        self._filepath: Optional[Path] = None
        self._event_count = 0

    def open(self) -> None:
        """Open log file for writing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = self.output_dir / f"events_{timestamp}.jsonl"

        self._file = open(self._filepath, "w", encoding="utf-8")

        # Write header event
        self._write_event({
            "type": "session_start",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "version": "2.0.0"
        })

        logger.info(f"Event log opened: {self._filepath}")

    def close(self) -> None:
        """Close log file."""
        if self._file:
            # Write footer event
            self._write_event({
                "type": "session_end",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "total_events": self._event_count
            })

            self._file.close()
            self._file = None
            logger.info(f"Event log closed: {self._filepath}")

    def log_state_event(self, event: StateEvent) -> None:
        """
        Log a state machine event.

        Args:
            event: State event to log
        """
        data = {
            "type": "state_change",
            "timestamp": event.timestamp,
            "datetime": datetime.fromtimestamp(event.timestamp).isoformat(),
            "event_type": event.event_type,
            "old_state": event.old_state.name,
            "new_state": event.new_state.name,
            "confidence": event.confidence,
            "message": event.message
        }

        if event.detection:
            data["detection"] = {
                "class": event.detection.class_name,
                "confidence": event.detection.confidence,
                "bbox": {
                    "x1": event.detection.x1,
                    "y1": event.detection.y1,
                    "x2": event.detection.x2,
                    "y2": event.detection.y2
                },
                "center": {
                    "x": event.detection.center[0],
                    "y": event.detection.center[1]
                }
            }

        self._write_event(data)

    def log_detection(
        self,
        frame_id: int,
        timestamp: float,
        detections: list,
        inference_time_ms: float
    ) -> None:
        """
        Log detection results (optional, for detailed logging).

        Args:
            frame_id: Frame number
            timestamp: Detection timestamp
            detections: List of detections
            inference_time_ms: Inference time
        """
        data = {
            "type": "detection",
            "timestamp": timestamp,
            "frame_id": frame_id,
            "inference_time_ms": inference_time_ms,
            "detection_count": len(detections),
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "bbox": [d.x1, d.y1, d.x2, d.y2]
                }
                for d in detections
            ]
        }

        self._write_event(data)

    def log_alert(self, event_type: str, confidence: float) -> None:
        """
        Log an alert trigger.

        Args:
            event_type: Type of alert event
            confidence: Detection confidence
        """
        self._write_event({
            "type": "alert",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "confidence": confidence
        })

    def log_custom(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a custom event.

        Args:
            event_type: Event type string
            data: Event data dictionary
        """
        event_data = {
            "type": event_type,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            **data
        }
        self._write_event(event_data)

    def _write_event(self, data: Dict[str, Any]) -> None:
        """Write event to log file."""
        if not self._file:
            return

        try:
            line = json.dumps(data, ensure_ascii=False)
            self._file.write(line + "\n")
            self._file.flush()
            self._event_count += 1
        except Exception as e:
            logger.error(f"Failed to write event: {e}")

    @property
    def filepath(self) -> Optional[Path]:
        """Get current log file path."""
        return self._filepath

    @property
    def event_count(self) -> int:
        """Get total events logged."""
        return self._event_count

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

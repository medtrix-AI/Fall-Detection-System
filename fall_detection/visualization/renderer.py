"""Visualization renderer for fall detection display."""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from ..detector.base import BoundingBox
from ..core.state_machine import FallState


# Color scheme (BGR)
COLORS = {
    "Fallen": (0, 0, 255),      # Red
    "Sitting": (0, 165, 255),   # Orange
    "Standing": (0, 255, 0),    # Green
    "default": (255, 255, 0),   # Cyan
}

STATE_COLORS = {
    FallState.NORMAL: (0, 255, 0),      # Green
    FallState.CANDIDATE: (0, 255, 255), # Yellow
    FallState.CONFIRMING: (0, 165, 255),# Orange
    FallState.ALERTED: (0, 0, 255),     # Red
    FallState.COOLDOWN: (255, 0, 255),  # Magenta
}


class VisualizationRenderer:
    """
    Renders visualization overlays on video frames.

    Features:
    - Bounding boxes with class labels
    - State indicator HUD
    - FPS counter
    - Alert overlay
    - Progress bars for confirmation/cooldown
    """

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        show_confidence: bool = True
    ):
        """
        Initialize renderer.

        Args:
            font_scale: Font scale for text
            thickness: Line thickness
            show_confidence: Show confidence values on boxes
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_confidence = show_confidence
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        detections: List[BoundingBox],
        state: FallState,
        fps: float = 0.0,
        time_in_state: float = 0.0,
        cooldown_remaining: float = 0.0,
        confirm_progress: float = 0.0,
        show_alert: bool = False
    ) -> np.ndarray:
        """
        Render all overlays on frame.

        Args:
            frame: Input BGR frame
            detections: List of detections to draw
            state: Current state machine state
            fps: Current FPS
            time_in_state: Time in current state
            cooldown_remaining: Remaining cooldown time
            confirm_progress: Confirmation progress (0-1)
            show_alert: Whether to show alert overlay

        Returns:
            Frame with overlays drawn
        """
        output = frame.copy()

        # Draw detections
        for det in detections:
            self._draw_detection(output, det)

        # Draw HUD
        self._draw_hud(output, state, fps, time_in_state)

        # Draw progress bar for confirming state
        if state == FallState.CONFIRMING and confirm_progress > 0:
            self._draw_progress_bar(
                output,
                confirm_progress,
                "Confirming",
                (0, 165, 255)
            )

        # Draw cooldown indicator
        if state == FallState.COOLDOWN and cooldown_remaining > 0:
            self._draw_cooldown(output, cooldown_remaining)

        # Draw alert overlay
        if show_alert:
            self._draw_alert_overlay(output)

        return output

    def _draw_detection(self, frame: np.ndarray, det: BoundingBox) -> None:
        """Draw a single detection box with label."""
        color = COLORS.get(det.class_name, COLORS["default"])

        # Draw box
        x1, y1, x2, y2 = det.as_xyxy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Build label
        if self.show_confidence:
            label = f"{det.class_name} {det.confidence:.0%}"
        else:
            label = det.class_name

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.thickness
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        state: FallState,
        fps: float,
        time_in_state: float
    ) -> None:
        """Draw heads-up display."""
        height, width = frame.shape[:2]

        # Background panel
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # State indicator
        state_color = STATE_COLORS.get(state, (255, 255, 255))
        state_text = f"State: {state.name}"
        cv2.putText(
            frame, state_text, (10, 30),
            self.font, 0.7, state_color, 2
        )

        # State indicator circle
        cv2.circle(frame, (width - 40, 30), 15, state_color, -1)

        # Time in state
        time_text = f"Time: {time_in_state:.1f}s"
        cv2.putText(
            frame, time_text, (10, 60),
            self.font, 0.5, (200, 200, 200), 1
        )

        # FPS counter
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (width - 100, 60),
            self.font, 0.5, (200, 200, 200), 1
        )

        # Model label
        cv2.putText(
            frame, "YOLOv11 Fall Detection", (width // 2 - 100, 30),
            self.font, 0.5, (200, 200, 200), 1
        )

    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        progress: float,
        label: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw a progress bar."""
        height, width = frame.shape[:2]

        bar_width = 200
        bar_height = 20
        x = (width - bar_width) // 2
        y = height - 60

        # Background
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (50, 50, 50),
            -1
        )

        # Progress fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(
            frame,
            (x, y),
            (x + fill_width, y + bar_height),
            color,
            -1
        )

        # Border
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (255, 255, 255),
            1
        )

        # Label
        text = f"{label}: {progress:.0%}"
        cv2.putText(
            frame, text,
            (x, y - 5),
            self.font, 0.5, (255, 255, 255), 1
        )

    def _draw_cooldown(self, frame: np.ndarray, remaining: float) -> None:
        """Draw cooldown indicator."""
        height, width = frame.shape[:2]

        text = f"Cooldown: {remaining:.1f}s"
        cv2.putText(
            frame, text,
            (width // 2 - 70, height - 30),
            self.font, 0.6, (255, 0, 255), 2
        )

    def _draw_alert_overlay(self, frame: np.ndarray) -> None:
        """Draw alert overlay."""
        height, width = frame.shape[:2]

        # Red border
        border_thickness = 10
        cv2.rectangle(
            frame,
            (0, 0),
            (width, height),
            (0, 0, 255),
            border_thickness
        )

        # Alert text
        alert_text = "FALL DETECTED!"
        (text_width, text_height), _ = cv2.getTextSize(
            alert_text, self.font, 1.5, 3
        )

        x = (width - text_width) // 2
        y = height // 2

        # Text background
        padding = 20
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            (0, 0, 255),
            -1
        )

        # Text
        cv2.putText(
            frame, alert_text,
            (x, y),
            self.font, 1.5, (255, 255, 255), 3
        )

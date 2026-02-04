"""Main fall detection application orchestrator."""

import time
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2

from ..config.settings import AppConfig
from ..detector.yolo_detector import YOLOFallDetector
from ..video.source import VideoSource, Frame
from ..video.recorder import ClipRecorder
from .state_machine import FallStateMachine, FallState, StateEvent
from ..alerts.manager import AlertManager
from ..alerts.sound import SoundAlertHandler
from ..alerts.visual import VisualAlertHandler
from ..logging_.event_logger import EventLogger
from ..visualization.renderer import VisualizationRenderer
from ..utils.fps_counter import FPSCounter

logger = logging.getLogger(__name__)


class FallDetectionApp:
    """
    Main application orchestrator for fall detection system.

    Coordinates all components: video input, detection, state machine,
    alerts, logging, and visualization.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the application.

        Args:
            config: Application configuration
        """
        self.config = config
        self._running = False
        self._setup_signal_handlers()

        # Run directory
        self._run_dir = self._create_run_directory()

        # Components (initialized in setup())
        self._detector: Optional[YOLOFallDetector] = None
        self._video_source: Optional[VideoSource] = None
        self._state_machine: Optional[FallStateMachine] = None
        self._alert_manager: Optional[AlertManager] = None
        self._event_logger: Optional[EventLogger] = None
        self._renderer: Optional[VisualizationRenderer] = None
        self._recorder: Optional[ClipRecorder] = None
        self._fps_counter: Optional[FPSCounter] = None

        # Display state
        self._show_alert_overlay = False
        self._alert_overlay_until = 0.0

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _create_run_directory(self) -> Path:
        """Create timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.config.output.output_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up fall detection application...")
        print("\n" + "=" * 60)
        print("Fall Detection System V2 - YOLOv11")
        print("=" * 60)

        try:
            # Detector
            print("Loading model from Hugging Face...")
            self._detector = YOLOFallDetector(self.config.model)
            self._detector.load_model()
            self._detector.warmup()
            print(f"  Model loaded: {self.config.model.repo_id}")
            print(f"  Device: {self._detector._device}")

            # Video source
            print(f"Opening video source: {self.config.video.source}...")
            self._video_source = VideoSource(self.config.video)
            if not self._video_source.open():
                print("  ERROR: Failed to open video source")
                return False
            print(f"  Resolution: {self.config.video.width}x{self.config.video.height}")

            # State machine
            self._state_machine = FallStateMachine(
                self.config.detection,
                on_state_change=self._on_state_change
            )
            print("  State machine initialized")

            # Alert manager
            self._alert_manager = AlertManager(self.config.alert)
            self._alert_manager.register_handler(SoundAlertHandler(self.config.alert))
            self._alert_manager.register_handler(VisualAlertHandler(self.config.alert))
            self._alert_manager.start()
            print(f"  Alert handlers: {self._alert_manager.handler_count}")

            # Event logger
            self._event_logger = EventLogger(self._run_dir)
            self._event_logger.open()
            print(f"  Event log: {self._event_logger.filepath}")

            # Visualization
            self._renderer = VisualizationRenderer()

            # Clip recorder
            if self.config.output.save_clips:
                clips_dir = self._run_dir / "clips"
                self._recorder = ClipRecorder(
                    output_dir=clips_dir,
                    pre_seconds=self.config.output.clip_pre_seconds,
                    post_seconds=self.config.output.clip_post_seconds,
                    fps=self._video_source.fps
                )
                self._recorder.enable()
                print(f"  Clip recorder: {clips_dir}")

            # FPS counter
            self._fps_counter = FPSCounter()

            print("=" * 60)
            print(f"Setup complete. Output: {self._run_dir}")
            print("=" * 60 + "\n")

            logger.info(f"Setup complete. Run directory: {self._run_dir}")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            print(f"\nERROR: Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_state_change(self, event: StateEvent) -> None:
        """Handle state machine events."""
        # Log event
        if self._event_logger:
            self._event_logger.log_state_event(event)

        logger.info(
            f"State change: {event.old_state.name} -> {event.new_state.name}: "
            f"{event.message}"
        )

        if event.event_type == "fall_confirmed":
            # Trigger alert
            if self._alert_manager:
                self._alert_manager.trigger_alert(event)

            # Show overlay
            self._show_alert_overlay = True
            self._alert_overlay_until = time.time() + 3.0

            # Trigger clip recording
            if self._recorder:
                self._recorder.trigger("fall")

            # Log alert
            if self._event_logger:
                self._event_logger.log_alert("fall_confirmed", event.confidence)

    def run(self) -> None:
        """Main application loop."""
        if not self._video_source or not self._video_source.is_open:
            logger.error("Video source not initialized")
            return

        self._running = True
        window_name = "Fall Detection - YOLOv11"

        if self.config.output.display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.config.video.width, self.config.video.height)

        print("Detection started. Press 'q' to quit, 'r' to reset state.\n")
        logger.info("Starting detection loop")

        try:
            for frame in self._video_source.frames():
                if not self._running:
                    break

                self._process_frame(frame, window_name)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup(window_name)

    def _process_frame(self, frame: Frame, window_name: str) -> None:
        """Process a single frame."""
        # Update FPS
        fps = self._fps_counter.tick()

        # Run detection
        result = self._detector.detect(frame.data)

        # Update state machine
        state_event = self._state_machine.update(result)

        # Add frame to recorder
        if self._recorder:
            self._recorder.add_frame(frame.data, frame.timestamp)

        # Check alert overlay timeout
        if self._show_alert_overlay and time.time() > self._alert_overlay_until:
            self._show_alert_overlay = False

        # Visualization
        if self.config.output.display:
            display_frame = self._renderer.render(
                frame=frame.data,
                detections=result.detections,
                state=self._state_machine.state,
                fps=fps,
                time_in_state=self._state_machine.time_in_state,
                cooldown_remaining=self._state_machine.cooldown_remaining,
                confirm_progress=self._state_machine.confirm_progress,
                show_alert=self._show_alert_overlay
            )

            cv2.imshow(window_name, display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False
            elif key == ord('r'):
                self._state_machine.reset()
                self._fps_counter.reset()
                print("State reset to NORMAL")

    def _cleanup(self, window_name: str) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")
        print("\nShutting down...")

        # Stop alert manager
        if self._alert_manager:
            self._alert_manager.stop()

        # Close loggers
        if self._event_logger:
            self._event_logger.close()

        # Release video source
        if self._video_source:
            self._video_source.release()

        # Close window
        if self.config.output.display:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        print(f"Output saved to: {self._run_dir}")
        logger.info(f"Cleanup complete. Output: {self._run_dir}")

    @property
    def is_running(self) -> bool:
        """Check if application is running."""
        return self._running

    def stop(self) -> None:
        """Stop the application."""
        self._running = False

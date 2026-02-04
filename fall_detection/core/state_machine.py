"""Fall detection state machine with multi-frame confirmation."""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, List, Deque
from collections import deque

from ..detector.base import DetectionResult, BoundingBox
from ..config.settings import DetectionConfig

logger = logging.getLogger(__name__)


class FallState(Enum):
    """Fall detection state machine states."""
    NORMAL = auto()      # Monitoring, no fall detected
    CANDIDATE = auto()   # Potential fall, validating
    CONFIRMING = auto()  # Fall detected, confirming persistence
    ALERTED = auto()     # Fall confirmed, alert triggered
    COOLDOWN = auto()    # Post-alert cooldown period


@dataclass
class StateEvent:
    """Event emitted on state transitions."""
    event_type: str  # "fall_candidate", "fall_confirmed", "recovery", "cooldown_end"
    timestamp: float
    old_state: FallState
    new_state: FallState
    detection: Optional[BoundingBox] = None
    confidence: float = 0.0
    message: str = ""


@dataclass
class DetectionHistoryEntry:
    """Single entry in detection history."""
    timestamp: float
    has_fall: bool
    confidence: float
    detection: Optional[BoundingBox] = None


class FallStateMachine:
    """
    State machine for fall detection with multi-frame confirmation.

    State Flow:
    NORMAL -> CANDIDATE: Fall detection above threshold
    CANDIDATE -> CONFIRMING: Sustained fall detection
    CANDIDATE -> NORMAL: No sustained detection (recovery)
    CONFIRMING -> ALERTED: Confirmed for confirm_duration_sec
    CONFIRMING -> NORMAL: Recovery during confirmation
    ALERTED -> COOLDOWN: Immediately after alert
    COOLDOWN -> NORMAL: After cooldown_duration_sec
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        on_state_change: Optional[Callable[[StateEvent], None]] = None
    ):
        """
        Initialize the state machine.

        Args:
            config: Detection configuration
            on_state_change: Callback for state changes
        """
        self.config = config or DetectionConfig()
        self.on_state_change = on_state_change

        # State tracking
        self._state = FallState.NORMAL
        self._state_entry_time: float = time.time()

        # Timing
        self._candidate_start: Optional[float] = None
        self._confirm_start: Optional[float] = None
        self._cooldown_start: Optional[float] = None

        # Detection history for multi-frame confirmation
        self._detection_history: Deque[DetectionHistoryEntry] = deque()

        # Current best detection
        self._current_detection: Optional[BoundingBox] = None

        # Event history
        self.events: List[StateEvent] = []

    @property
    def state(self) -> FallState:
        """Get current state."""
        return self._state

    @property
    def state_name(self) -> str:
        """Get current state name."""
        return self._state.name

    @property
    def time_in_state(self) -> float:
        """Get time spent in current state."""
        return time.time() - self._state_entry_time

    @property
    def cooldown_remaining(self) -> float:
        """Get remaining cooldown time."""
        if self._state != FallState.COOLDOWN or not self._cooldown_start:
            return 0.0
        elapsed = time.time() - self._cooldown_start
        return max(0.0, self.config.cooldown_duration_sec - elapsed)

    @property
    def confirm_progress(self) -> float:
        """Get progress toward confirmation (0.0 to 1.0)."""
        if self._state != FallState.CONFIRMING or not self._confirm_start:
            return 0.0
        elapsed = time.time() - self._confirm_start
        return min(1.0, elapsed / self.config.confirm_duration_sec)

    @property
    def current_detection(self) -> Optional[BoundingBox]:
        """Get current fall detection if any."""
        return self._current_detection

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._transition_to(FallState.NORMAL, "Manual reset")
        self._detection_history.clear()
        self._current_detection = None
        self._candidate_start = None
        self._confirm_start = None
        self._cooldown_start = None

    def update(self, result: DetectionResult) -> Optional[StateEvent]:
        """
        Update state machine with new detection result.

        Args:
            result: Detection result from detector

        Returns:
            StateEvent if state changed, None otherwise
        """
        timestamp = result.timestamp

        # Prune old history
        self._prune_history(timestamp)

        # Find best fall detection
        best_fall = result.get_best_fallen()

        has_fall = (
            best_fall is not None and
            best_fall.confidence >= self.config.fall_confidence_threshold
        )

        # Add to history
        self._detection_history.append(DetectionHistoryEntry(
            timestamp=timestamp,
            has_fall=has_fall,
            confidence=best_fall.confidence if best_fall else 0.0,
            detection=best_fall
        ))

        if has_fall:
            self._current_detection = best_fall

        # Process state
        return self._process_state(timestamp, has_fall)

    def _prune_history(self, current_time: float) -> None:
        """Remove entries older than detection window."""
        cutoff = current_time - self.config.detection_window_sec
        while self._detection_history and self._detection_history[0].timestamp < cutoff:
            self._detection_history.popleft()

    def _count_recent_falls(self) -> int:
        """Count fall detections in recent history."""
        return sum(1 for e in self._detection_history if e.has_fall)

    def _get_average_confidence(self) -> float:
        """Get average confidence of recent fall detections."""
        fall_entries = [e for e in self._detection_history if e.has_fall]
        if not fall_entries:
            return 0.0
        return sum(e.confidence for e in fall_entries) / len(fall_entries)

    def _process_state(self, timestamp: float, has_fall: bool) -> Optional[StateEvent]:
        """Process state transitions based on current detection."""
        if self._state == FallState.NORMAL:
            return self._handle_normal(timestamp, has_fall)
        elif self._state == FallState.CANDIDATE:
            return self._handle_candidate(timestamp, has_fall)
        elif self._state == FallState.CONFIRMING:
            return self._handle_confirming(timestamp, has_fall)
        elif self._state == FallState.ALERTED:
            return self._handle_alerted(timestamp)
        elif self._state == FallState.COOLDOWN:
            return self._handle_cooldown(timestamp)
        return None

    def _handle_normal(self, timestamp: float, has_fall: bool) -> Optional[StateEvent]:
        """Handle NORMAL state - look for fall candidates."""
        if has_fall:
            self._candidate_start = timestamp
            return self._transition_to(
                FallState.CANDIDATE,
                f"Fall detected (conf={self._current_detection.confidence:.2f})"
            )
        return None

    def _handle_candidate(self, timestamp: float, has_fall: bool) -> Optional[StateEvent]:
        """Handle CANDIDATE state - validate detection."""
        if self._candidate_start is None:
            self._candidate_start = timestamp

        time_in_candidate = timestamp - self._candidate_start
        consecutive_falls = self._count_recent_falls()

        # Check for sustained detection
        if consecutive_falls >= self.config.min_consecutive_detections:
            if time_in_candidate >= self.config.candidate_validation_sec:
                self._candidate_start = None
                self._confirm_start = timestamp
                return self._transition_to(
                    FallState.CONFIRMING,
                    f"Fall validated ({consecutive_falls} detections)"
                )

        # Check for recovery
        if not has_fall and time_in_candidate > self.config.recovery_threshold_sec:
            self._candidate_start = None
            self._current_detection = None
            return self._transition_to(FallState.NORMAL, "Recovery during validation")

        return None

    def _handle_confirming(self, timestamp: float, has_fall: bool) -> Optional[StateEvent]:
        """Handle CONFIRMING state - wait for confirmation period."""
        if self._confirm_start is None:
            self._confirm_start = timestamp

        time_confirming = timestamp - self._confirm_start

        # Check for recovery (multiple consecutive non-fall frames)
        if not has_fall:
            recent_entries = list(self._detection_history)[-5:]
            consecutive_no_fall = sum(1 for e in recent_entries if not e.has_fall)
            if consecutive_no_fall >= 3:
                self._confirm_start = None
                self._current_detection = None
                return self._transition_to(FallState.NORMAL, "Recovery during confirmation")

        # Check if confirmed
        if time_confirming >= self.config.confirm_duration_sec:
            avg_conf = self._get_average_confidence()
            self._confirm_start = None
            return self._transition_to(
                FallState.ALERTED,
                f"FALL CONFIRMED (avg_conf={avg_conf:.2f}, duration={time_confirming:.1f}s)"
            )

        return None

    def _handle_alerted(self, timestamp: float) -> Optional[StateEvent]:
        """Handle ALERTED state - transition to cooldown."""
        self._cooldown_start = timestamp
        return self._transition_to(
            FallState.COOLDOWN,
            f"Entering cooldown ({self.config.cooldown_duration_sec}s)"
        )

    def _handle_cooldown(self, timestamp: float) -> Optional[StateEvent]:
        """Handle COOLDOWN state - wait for cooldown to end."""
        if self._cooldown_start is None:
            self._cooldown_start = timestamp

        elapsed = timestamp - self._cooldown_start
        if elapsed >= self.config.cooldown_duration_sec:
            self._cooldown_start = None
            self._current_detection = None
            return self._transition_to(FallState.NORMAL, "Cooldown complete")

        return None

    def _transition_to(self, new_state: FallState, message: str) -> StateEvent:
        """Transition to a new state and emit event."""
        old_state = self._state
        self._state = new_state
        self._state_entry_time = time.time()

        event = StateEvent(
            event_type=self._get_event_type(old_state, new_state),
            timestamp=time.time(),
            old_state=old_state,
            new_state=new_state,
            detection=self._current_detection,
            confidence=self._current_detection.confidence if self._current_detection else 0.0,
            message=message
        )

        self.events.append(event)
        logger.info(f"State: {old_state.name} -> {new_state.name}: {message}")

        if self.on_state_change:
            self.on_state_change(event)

        return event

    def _get_event_type(self, old: FallState, new: FallState) -> str:
        """Determine event type from state transition."""
        if new == FallState.CANDIDATE:
            return "fall_candidate"
        elif new == FallState.CONFIRMING:
            return "fall_validating"
        elif new == FallState.ALERTED:
            return "fall_confirmed"
        elif new == FallState.COOLDOWN:
            return "cooldown_start"
        elif new == FallState.NORMAL:
            if old == FallState.COOLDOWN:
                return "cooldown_end"
            return "recovery"
        return "unknown"

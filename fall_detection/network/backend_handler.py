"""Backend alert handler — sends fall events to the Medtrix API."""

import logging
import threading
from typing import Optional

from ..alerts.manager import BaseAlertHandler, AlertContext
from .client import BackendClient

logger = logging.getLogger(__name__)


class BackendAlertHandler(BaseAlertHandler):
    """
    Alert handler that POSTs fall events to the backend edge API.

    Registered with AlertManager via register_handler(). Runs in the
    AlertManager's worker thread — BackendClient handles retry/spool.
    """

    def __init__(self, client: BackendClient, location: Optional[str] = None):
        self._client = client
        self._location = location
        self._last_event_id: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "BackendAlert"

    def is_available(self) -> bool:
        return True  # always available — spools offline if backend unreachable

    def send_alert(self, context: AlertContext) -> bool:
        event = context.event
        confidence = event.confidence

        # Build metadata from detection bounding box
        metadata: dict = {
            "event_type": event.event_type,
            "state_transition": f"{event.old_state.name} -> {event.new_state.name}",
            "message": event.message,
        }

        if event.detection:
            bbox = event.detection
            metadata["bbox"] = {
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
            }
            metadata["class_name"] = bbox.class_name
            metadata["class_id"] = bbox.class_id

        result = self._client.post_event(
            event_type="fall_detected",
            confidence=confidence,
            location=self._location,
            metadata=metadata,
        )

        if result and "event_id" in result:
            with self._lock:
                self._last_event_id = result["event_id"]
            logger.info("backend_alert_sent event_id=%s conf=%.2f", result["event_id"], confidence)
            return True

        # Even if result is None (spooled), we consider it "sent" — will be replayed
        if result is None:
            logger.info("backend_alert_spooled conf=%.2f", confidence)
            return True

        return False

    @property
    def last_event_id(self) -> Optional[str]:
        """Get the event_id from the last successful POST (for clip association)."""
        with self._lock:
            return self._last_event_id

"""Heartbeat service — daemon thread sending periodic heartbeats and replaying spooled events."""

import logging
import threading
from typing import Optional

from .client import BackendClient

logger = logging.getLogger(__name__)


class HeartbeatService:
    """
    Daemon thread that sends POST /edge/heartbeat at a fixed interval.

    On each successful heartbeat, replays any offline-spooled events.
    Uses interruptible sleep (0.5s increments) for clean shutdown.
    """

    def __init__(
        self,
        client: BackendClient,
        interval_sec: int = 300,
    ):
        self._client = client
        self._interval = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the heartbeat daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="HeartbeatService",
        )
        self._thread.start()
        logger.info("HeartbeatService started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the heartbeat thread to stop and wait."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("HeartbeatService stopped")

    def _run(self) -> None:
        """Main heartbeat loop."""
        # Send an initial heartbeat immediately
        self._send_heartbeat()

        while not self._stop_event.is_set():
            # Interruptible sleep in 0.5s increments
            for _ in range(self._interval * 2):
                if self._stop_event.is_set():
                    return
                self._stop_event.wait(0.5)

            if self._stop_event.is_set():
                return

            self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat and replay spool on success."""
        result = self._client.post_heartbeat()
        if result is not None:
            logger.debug("Heartbeat sent successfully")
            # Connectivity confirmed — drain offline spool
            replayed = self._client.replay_spool()
            if replayed:
                logger.info("Replayed %d spooled events after heartbeat", replayed)
        else:
            logger.warning("Heartbeat failed — backend may be unreachable")

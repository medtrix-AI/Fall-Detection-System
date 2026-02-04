"""Alert manager with thread-safe, non-blocking delivery."""

import time
import threading
import queue
import logging
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.state_machine import StateEvent
from ..config.settings import AlertConfig

logger = logging.getLogger(__name__)


@dataclass
class AlertContext:
    """Context for alert delivery."""
    event: StateEvent
    frame_path: Optional[str] = None
    clip_path: Optional[str] = None


class BaseAlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging."""
        pass

    @abstractmethod
    def send_alert(self, context: AlertContext) -> bool:
        """
        Send an alert.

        Args:
            context: Alert context with event details

        Returns:
            True if alert was sent successfully
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this handler is available/configured."""
        pass


class AlertManager:
    """
    Manages multiple alert handlers with thread-safe, non-blocking delivery.

    Features:
    - Multiple simultaneous alert channels
    - Cooldown to prevent alert spam
    - Rate limiting
    - Async delivery to prevent blocking video loop
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self._handlers: List[BaseAlertHandler] = []

        # Cooldown tracking
        self._last_alert_time: float = 0
        self._alerts_this_minute: int = 0
        self._minute_start: float = time.time()

        # Async delivery
        self._alert_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Thread safety
        self._lock = threading.Lock()

    def register_handler(self, handler: BaseAlertHandler) -> None:
        """Register an alert handler."""
        if handler.is_available():
            self._handlers.append(handler)
            logger.info(f"Registered alert handler: {handler.name}")
        else:
            logger.warning(f"Alert handler not available: {handler.name}")

    def start(self) -> None:
        """Start the alert manager worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AlertManagerWorker"
        )
        self._worker_thread.start()
        logger.info("Alert manager started")

    def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        self._alert_queue.put(None)  # Sentinel to unblock
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        logger.info("Alert manager stopped")

    def trigger_alert(
        self,
        event: StateEvent,
        frame_path: Optional[str] = None,
        clip_path: Optional[str] = None
    ) -> bool:
        """
        Trigger an alert if cooldown allows.

        Args:
            event: The fall detection event
            frame_path: Optional path to screenshot
            clip_path: Optional path to video clip

        Returns:
            True if alert was queued, False if blocked by cooldown/rate limit
        """
        with self._lock:
            current_time = time.time()

            # Check cooldown
            if current_time - self._last_alert_time < self.config.alert_cooldown_sec:
                logger.debug("Alert blocked by cooldown")
                return False

            # Check rate limit
            if current_time - self._minute_start >= 60:
                self._minute_start = current_time
                self._alerts_this_minute = 0

            if self._alerts_this_minute >= self.config.max_alerts_per_minute:
                logger.debug("Alert blocked by rate limit")
                return False

            # Queue alert
            context = AlertContext(
                event=event,
                frame_path=frame_path,
                clip_path=clip_path
            )

            self._alert_queue.put(context)
            self._last_alert_time = current_time
            self._alerts_this_minute += 1

            logger.info("Alert queued for delivery")
            return True

    def _worker_loop(self) -> None:
        """Background worker for async alert delivery."""
        while self._running:
            try:
                context = self._alert_queue.get(timeout=0.5)
                if context is None:
                    continue

                self._deliver_alert(context)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in alert worker: {e}")

    def _deliver_alert(self, context: AlertContext) -> None:
        """Deliver alert to all registered handlers."""
        for handler in self._handlers:
            try:
                success = handler.send_alert(context)
                if success:
                    logger.debug(f"Alert delivered via {handler.name}")
                else:
                    logger.warning(f"Alert delivery failed for {handler.name}")
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {e}")

    @property
    def handler_count(self) -> int:
        """Get number of registered handlers."""
        return len(self._handlers)

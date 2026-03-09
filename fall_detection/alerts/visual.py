"""Visual alert handler — console-based."""

import logging
from .manager import BaseAlertHandler, AlertContext
from ..config.settings import AlertConfig

logger = logging.getLogger(__name__)


class VisualAlertHandler(BaseAlertHandler):
    """Visual alert via console banner.

    The on-screen overlay (red border + text) is already handled by the
    OpenCV renderer, so this handler prints a prominent console banner
    instead of spawning a tkinter popup (which crashes on macOS when
    called from a non-main thread).
    """

    def __init__(self, config: AlertConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "VisualAlert"

    def is_available(self) -> bool:
        return self.config.enable_popup

    def send_alert(self, context: AlertContext) -> bool:
        try:
            confidence = context.event.confidence
            lines = [
                "",
                "!" * 50,
                "!!!        FALL DETECTED        !!!",
                "!" * 50,
                f"  Confidence : {confidence:.1%}",
            ]

            if context.event.detection:
                det = context.event.detection
                lines.append(
                    f"  Location   : ({int(det.center[0])}, {int(det.center[1])})"
                )

            lines.append("!" * 50)
            lines.append("")

            print("\n".join(lines), flush=True)
            logger.debug("Visual alert shown (console)")
            return True

        except Exception as e:
            logger.error(f"Visual alert error: {e}")
            return False

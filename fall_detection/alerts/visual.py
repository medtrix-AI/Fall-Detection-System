"""Visual popup alert handler."""

import threading
import logging
from .manager import BaseAlertHandler, AlertContext
from ..config.settings import AlertConfig

logger = logging.getLogger(__name__)


class VisualAlertHandler(BaseAlertHandler):
    """Visual popup alert using tkinter."""

    def __init__(self, config: AlertConfig):
        """
        Initialize visual alert handler.

        Args:
            config: Alert configuration
        """
        self.config = config
        self._tkinter_available = False

        try:
            import tkinter
            self._tkinter_available = True
        except ImportError:
            logger.warning("tkinter not available for visual alerts")

    @property
    def name(self) -> str:
        """Handler name."""
        return "VisualAlert"

    def is_available(self) -> bool:
        """Check if visual alerts are available and enabled."""
        return self._tkinter_available and self.config.enable_popup

    def send_alert(self, context: AlertContext) -> bool:
        """
        Show popup alert.

        Args:
            context: Alert context

        Returns:
            True if popup was shown
        """
        # Run popup in separate thread to not block
        thread = threading.Thread(
            target=self._show_popup,
            args=(context,),
            daemon=True,
            name="VisualAlertPopup"
        )
        thread.start()
        return True

    def _show_popup(self, context: AlertContext) -> None:
        """Show the popup window."""
        try:
            import tkinter as tk
            from tkinter import messagebox

            # Create hidden root window
            root = tk.Tk()
            root.withdraw()

            # Make it topmost
            root.attributes('-topmost', True)
            root.update()

            # Build message
            confidence = context.event.confidence
            message = f"A fall has been detected!\n\nConfidence: {confidence:.1%}"

            if context.event.detection:
                det = context.event.detection
                message += f"\nLocation: ({int(det.center[0])}, {int(det.center[1])})"

            # Show warning dialog
            messagebox.showwarning(
                "FALL ALERT",
                message,
                parent=root
            )

            root.destroy()
            logger.debug("Visual alert shown")

        except Exception as e:
            logger.error(f"Visual alert error: {e}")

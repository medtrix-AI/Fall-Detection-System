"""Sound alert handler."""

import logging
import platform
from .manager import BaseAlertHandler, AlertContext
from ..config.settings import AlertConfig

logger = logging.getLogger(__name__)


class SoundAlertHandler(BaseAlertHandler):
    """Sound alert using system beep."""

    def __init__(self, config: AlertConfig):
        """
        Initialize sound alert handler.

        Args:
            config: Alert configuration
        """
        self.config = config
        self._available = False
        self._system = platform.system()

        # Check availability
        if self._system == "Windows":
            try:
                import winsound
                self._available = True
            except ImportError:
                pass
        else:
            # On other systems, we'll use terminal bell
            self._available = True

    @property
    def name(self) -> str:
        """Handler name."""
        return "SoundAlert"

    def is_available(self) -> bool:
        """Check if sound alerts are available and enabled."""
        return self._available and self.config.enable_sound

    def send_alert(self, context: AlertContext) -> bool:
        """
        Play alert sound.

        Args:
            context: Alert context

        Returns:
            True if sound was played
        """
        try:
            if self._system == "Windows":
                import winsound
                # Play beep
                winsound.Beep(
                    self.config.sound_frequency,
                    self.config.sound_duration_ms
                )
                # Play a second beep for emphasis
                winsound.Beep(
                    self.config.sound_frequency + 200,
                    self.config.sound_duration_ms
                )
            else:
                # Terminal bell on other systems
                print('\a', end='', flush=True)

            logger.debug("Sound alert played")
            return True

        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
            return False

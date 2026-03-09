"""Sound alert handler — cross-platform."""

import logging
import os
import platform
import shutil
import subprocess
from .manager import BaseAlertHandler, AlertContext
from ..config.settings import AlertConfig

logger = logging.getLogger(__name__)

# macOS system sounds (checked in order of preference)
_MACOS_SOUNDS = [
    "/System/Library/Sounds/Sosumi.aiff",
    "/System/Library/Sounds/Glass.aiff",
    "/System/Library/Sounds/Ping.aiff",
    "/System/Library/Sounds/Hero.aiff",
]

# Linux freedesktop alert sounds
_LINUX_SOUNDS = [
    "/usr/share/sounds/freedesktop/stereo/bell.oga",
    "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
    "/usr/share/sounds/gnome/default/alerts/bark.ogg",
]


class SoundAlertHandler(BaseAlertHandler):
    """Sound alert using platform-native playback."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self._system = platform.system()
        self._sound_method: str | None = None
        self._sound_file: str | None = None
        self._player_cmd: str | None = None

        self._detect_method()

    def _detect_method(self) -> None:
        """Detect the best available sound playback method."""
        if self._system == "Windows":
            try:
                import winsound  # noqa: F401
                self._sound_method = "winsound"
            except ImportError:
                pass

        elif self._system == "Darwin":
            if shutil.which("afplay"):
                self._player_cmd = "afplay"
                for path in _MACOS_SOUNDS:
                    if os.path.isfile(path):
                        self._sound_file = path
                        break
                if self._sound_file:
                    self._sound_method = "afplay"

        else:  # Linux / other
            for cmd in ("paplay", "aplay"):
                if shutil.which(cmd):
                    self._player_cmd = cmd
                    break
            if self._player_cmd:
                for path in _LINUX_SOUNDS:
                    if os.path.isfile(path):
                        self._sound_file = path
                        break
                if self._sound_file:
                    self._sound_method = self._player_cmd

        # Fallback: terminal bell is always available
        if self._sound_method is None:
            self._sound_method = "bell"

        logger.debug(
            "Sound method: %s (file=%s)", self._sound_method, self._sound_file
        )

    @property
    def name(self) -> str:
        return "SoundAlert"

    def is_available(self) -> bool:
        return self._sound_method is not None and self.config.enable_sound

    def send_alert(self, context: AlertContext) -> bool:
        try:
            if self._sound_method == "winsound":
                self._play_winsound()
            elif self._sound_method in ("afplay", "paplay", "aplay"):
                self._play_subprocess()
            else:
                print("\a", end="", flush=True)

            logger.debug("Sound alert played via %s", self._sound_method)
            return True

        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
            return False

    def _play_winsound(self) -> None:
        import winsound
        winsound.Beep(self.config.sound_frequency, self.config.sound_duration_ms)
        winsound.Beep(
            self.config.sound_frequency + 200, self.config.sound_duration_ms
        )

    def _play_subprocess(self) -> None:
        """Play sound file via system command (non-blocking)."""
        subprocess.Popen(
            [self._player_cmd, self._sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

"""Clip uploader — daemon thread that watches for new MP4 files and uploads them."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .client import BackendClient

logger = logging.getLogger(__name__)


class ClipUploader:
    """
    Daemon thread that polls a clips directory for new .mp4 files
    and uploads them to the backend via POST /edge/clips.

    Waits for files to settle (no writes for 3s) before uploading.
    Associates clips with the most recent event_id from BackendAlertHandler.
    """

    SETTLE_SECONDS = 3.0
    POLL_INTERVAL = 2.0

    def __init__(
        self,
        client: BackendClient,
        clips_dir: Path,
        delete_after_upload: bool = False,
    ):
        self._client = client
        self._clips_dir = clips_dir
        self._delete_after = delete_after_upload
        self._event_id: Optional[str] = None
        self._event_id_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._uploaded: set[str] = set()

    def set_event_id(self, event_id: str) -> None:
        """Set the event_id to associate with the next uploaded clip."""
        with self._event_id_lock:
            self._event_id = event_id
        logger.debug("ClipUploader event_id set to %s", event_id)

    def start(self) -> None:
        """Start the clip uploader daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ClipUploader",
        )
        self._thread.start()
        logger.info("ClipUploader started (dir=%s)", self._clips_dir)

    def stop(self) -> None:
        """Signal the uploader thread to stop and wait."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("ClipUploader stopped")

    def _run(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.POLL_INTERVAL)
            if self._stop_event.is_set():
                return

            self._scan_and_upload()

    def _scan_and_upload(self) -> None:
        """Scan clips directory for settled .mp4 files and upload them."""
        if not self._clips_dir.exists():
            return

        for mp4 in self._clips_dir.glob("*.mp4"):
            if str(mp4) in self._uploaded:
                continue

            # Check if file has settled (no modification in SETTLE_SECONDS)
            try:
                mtime = os.path.getmtime(mp4)
                if time.time() - mtime < self.SETTLE_SECONDS:
                    continue  # still being written
            except OSError:
                continue

            self._upload_file(mp4)

    def _upload_file(self, clip_path: Path) -> None:
        """Upload a single clip file."""
        with self._event_id_lock:
            event_id = self._event_id

        if not event_id:
            logger.warning("No event_id available for clip %s — skipping", clip_path.name)
            return

        # Estimate duration from filename or default
        duration = self._estimate_duration(clip_path)

        result = self._client.upload_clip(
            event_id=event_id,
            event_type="fall_detected",
            clip_path=clip_path,
            duration=duration,
        )

        if result is not None:
            self._uploaded.add(str(clip_path))
            logger.info("Clip uploaded: %s", clip_path.name)
            if self._delete_after:
                try:
                    clip_path.unlink()
                    logger.debug("Deleted uploaded clip: %s", clip_path.name)
                except OSError as e:
                    logger.warning("Failed to delete clip %s: %s", clip_path.name, e)
        else:
            logger.warning("Clip upload failed: %s — will retry next cycle", clip_path.name)

    @staticmethod
    def _estimate_duration(clip_path: Path) -> Optional[int]:
        """Try to estimate clip duration from file size (rough heuristic)."""
        try:
            size_mb = clip_path.stat().st_size / (1024 * 1024)
            # Rough estimate: ~1MB per 10 seconds at typical edge quality
            return max(1, int(size_mb * 10))
        except OSError:
            return None

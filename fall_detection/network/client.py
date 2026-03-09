"""Thread-safe HTTP client for backend communication with offline spool."""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BackendClient:
    """
    HTTP client for the Medtrix backend edge API.

    Features:
    - x-device-key auth header on every request
    - urllib3 retry with exponential backoff on 5xx
    - Offline spool: failed POSTs saved to JSONL, replayed on connectivity
    - Thread-safe
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        device_id: str,
        spool_dir: Path,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.device_id = device_id
        self.timeout = timeout
        self._spool_path = spool_dir / "offline_spool.jsonl"
        self._spool_lock = threading.Lock()

        # Build session with retry
        self._session = requests.Session()
        self._session.headers.update({
            "x-device-key": api_key,
            "Content-Type": "application/json",
        })

        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        logger.info("BackendClient initialised (url=%s, device=%s)", base_url, device_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_event(
        self,
        event_type: str,
        confidence: float,
        location: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """POST /edge/events — returns response data or None on failure."""
        payload = {
            "type": event_type,
            "confidence": confidence,
            "device_id": self.device_id,
            "location": location,
            "metadata": metadata,
        }
        return self._post("/edge/events", payload, spool_on_fail=True)

    def post_heartbeat(self, metadata: Optional[dict] = None) -> Optional[dict]:
        """POST /edge/heartbeat."""
        payload = {
            "device_id": self.device_id,
            "metadata": metadata,
        }
        return self._post("/edge/heartbeat", payload, spool_on_fail=False)

    def upload_clip(
        self,
        event_id: str,
        event_type: str,
        clip_path: Path,
        duration: Optional[int] = None,
    ) -> Optional[dict]:
        """POST /edge/clips — multipart upload."""
        try:
            with open(clip_path, "rb") as f:
                files = {"file": (clip_path.name, f, "video/mp4")}
                data = {
                    "event_id": event_id,
                    "event_type": event_type,
                }
                if duration is not None:
                    data["duration_seconds"] = str(duration)

                # Override session headers: keep auth but remove Content-Type
                # so requests can set multipart/form-data with boundary automatically
                upload_headers = {
                    "x-device-key": self.api_key,
                    "Content-Type": None,
                }
                resp = self._session.post(
                    f"{self.base_url}/edge/clips",
                    files=files,
                    data=data,
                    headers=upload_headers,
                    timeout=self.timeout * 3,  # longer for uploads
                )

            resp.raise_for_status()
            result = resp.json().get("data", {})
            logger.info("clip_uploaded event_id=%s clip=%s", event_id, clip_path.name)
            return result
        except Exception as e:
            logger.error("clip_upload_failed clip=%s: %s", clip_path.name, e)
            return None

    def replay_spool(self) -> int:
        """Replay offline-spooled events. Returns count of successfully replayed."""
        with self._spool_lock:
            if not self._spool_path.exists():
                return 0

            lines = self._spool_path.read_text().strip().splitlines()
            if not lines:
                return 0

            logger.info("Replaying %d spooled events", len(lines))
            remaining = []
            replayed = 0

            for line in lines:
                try:
                    entry = json.loads(line)
                    url = entry["url"]
                    payload = entry["payload"]
                    resp = self._session.post(
                        url,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code < 500:
                        replayed += 1
                    else:
                        remaining.append(line)
                except Exception:
                    remaining.append(line)

            # Re-write only what failed
            if remaining:
                self._spool_path.write_text("\n".join(remaining) + "\n")
            else:
                self._spool_path.unlink(missing_ok=True)

            if replayed:
                logger.info("Replayed %d/%d spooled events", replayed, len(lines))
            return replayed

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(
        self,
        path: str,
        payload: dict,
        spool_on_fail: bool = False,
    ) -> Optional[dict]:
        """POST JSON to the backend. Optionally spool on failure."""
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)

            if resp.status_code == 401:
                logger.warning("backend_auth_failed (401) — check API key")
                return None

            resp.raise_for_status()
            return resp.json().get("data", {})

        except requests.RequestException as e:
            logger.warning("backend_request_failed path=%s: %s", path, e)
            if spool_on_fail:
                self._spool(url, payload)
            return None

    def _spool(self, url: str, payload: dict) -> None:
        """Append a failed request to the offline spool file."""
        entry = {
            "url": url,
            "payload": payload,
            "spooled_at": time.time(),
        }
        with self._spool_lock:
            self._spool_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._spool_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        logger.info("Event spooled for offline replay (%s)", url)

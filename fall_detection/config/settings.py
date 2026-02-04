"""Configuration settings using dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    repo_id: str = "melihuzunoglu/human-fall-detection"
    filename: str = "best.pt"
    input_size: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class VideoConfig:
    """Video source configuration."""
    source: str = "webcam"  # "webcam" or "video"
    video_path: Optional[str] = None
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps_target: int = 30
    buffer_size: int = 3


@dataclass
class DetectionConfig:
    """Fall detection state machine configuration."""
    candidate_validation_sec: float = 0.3
    confirm_duration_sec: float = 1.5
    cooldown_duration_sec: float = 15.0
    fall_confidence_threshold: float = 0.6
    min_consecutive_detections: int = 3
    detection_window_sec: float = 2.0
    recovery_threshold_sec: float = 0.5


@dataclass
class AlertConfig:
    """Alert system configuration."""
    enable_sound: bool = True
    enable_popup: bool = True
    alert_cooldown_sec: float = 30.0
    max_alerts_per_minute: int = 2
    sound_frequency: int = 1000
    sound_duration_ms: int = 500


@dataclass
class OutputConfig:
    """Output configuration."""
    display: bool = True
    save_clips: bool = True
    output_dir: str = "runs"
    clip_pre_seconds: float = 5.0
    clip_post_seconds: float = 5.0


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            model=ModelConfig(**data.get("model", {})),
            video=VideoConfig(**data.get("video", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            alert=AlertConfig(**data.get("alert", {})),
            output=OutputConfig(**data.get("output", {})),
            log_level=data.get("log_level", "INFO"),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": {
                "repo_id": self.model.repo_id,
                "filename": self.model.filename,
                "confidence_threshold": self.model.confidence_threshold,
                "device": self.model.device,
            },
            "video": {
                "source": self.video.source,
                "camera_index": self.video.camera_index,
                "video_path": self.video.video_path,
                "width": self.video.width,
                "height": self.video.height,
            },
            "detection": {
                "confirm_duration_sec": self.detection.confirm_duration_sec,
                "cooldown_duration_sec": self.detection.cooldown_duration_sec,
                "min_consecutive_detections": self.detection.min_consecutive_detections,
            },
            "alert": {
                "enable_sound": self.alert.enable_sound,
                "enable_popup": self.alert.enable_popup,
            },
            "output": {
                "display": self.output.display,
                "save_clips": self.output.save_clips,
                "output_dir": self.output.output_dir,
            },
            "log_level": self.log_level,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

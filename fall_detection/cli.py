"""Command-line interface for fall detection system."""

import argparse
import sys
import logging
from pathlib import Path

from .config.settings import AppConfig
from .core.app import FallDetectionApp


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Reduce noise from external libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fall Detection System V2 - YOLOv11",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )

    # Video source
    source_group = parser.add_argument_group("Video Source")
    source_group.add_argument(
        "--source",
        type=str,
        choices=["webcam", "video"],
        default="webcam",
        help="Video source type"
    )
    source_group.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to video file (required if source=video)"
    )
    source_group.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for webcam"
    )
    source_group.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width"
    )
    source_group.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height"
    )

    # Detection
    detect_group = parser.add_argument_group("Detection")
    detect_group.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    detect_group.add_argument(
        "--fall-threshold",
        type=float,
        default=0.6,
        help="Fall confirmation confidence threshold"
    )
    detect_group.add_argument(
        "--confirm-sec",
        type=float,
        default=1.5,
        help="Confirmation duration in seconds"
    )
    detect_group.add_argument(
        "--cooldown-sec",
        type=float,
        default=15.0,
        help="Cooldown duration after alert"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    output_group.add_argument(
        "--save-clips",
        action="store_true",
        default=True,
        help="Save video clips around detection events"
    )
    output_group.add_argument(
        "--no-clips",
        action="store_true",
        help="Disable clip recording"
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for logs and recordings"
    )

    # Alerts
    alert_group = parser.add_argument_group("Alerts")
    alert_group.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable sound alerts"
    )
    alert_group.add_argument(
        "--no-popup",
        action="store_true",
        help="Disable visual popup alerts"
    )

    # Other
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build configuration from arguments."""
    # Start with defaults or loaded config
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = AppConfig.from_yaml(config_path)
        else:
            print(f"Warning: Config file not found: {config_path}")
            config = AppConfig()
    else:
        config = AppConfig()

    # Override with CLI arguments
    config.video.source = args.source
    config.video.video_path = args.video_path
    config.video.camera_index = args.camera
    config.video.width = args.width
    config.video.height = args.height

    config.model.confidence_threshold = args.confidence
    config.model.device = args.device

    config.detection.fall_confidence_threshold = args.fall_threshold
    config.detection.confirm_duration_sec = args.confirm_sec
    config.detection.cooldown_duration_sec = args.cooldown_sec

    config.output.display = not args.no_display
    config.output.save_clips = args.save_clips and not args.no_clips
    config.output.output_dir = args.output_dir

    config.alert.enable_sound = not args.no_sound
    config.alert.enable_popup = not args.no_popup

    config.log_level = args.log_level

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(args.log_level)

    # Validate arguments
    if args.source == "video" and not args.video_path:
        print("Error: --video-path is required when --source=video")
        return 1

    config = build_config(args)

    app = FallDetectionApp(config)

    if not app.setup():
        print("Failed to initialize application")
        return 1

    app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())

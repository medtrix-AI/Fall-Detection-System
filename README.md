# Fall Detection System V2

Real-time fall detection using YOLOv11 from Hugging Face.

## Features

- **Pre-trained Model**: Uses `melihuzunoglu/human-fall-detection` from Hugging Face (no training required)
- **Real-time Detection**: Detects Fallen, Sitting, and Standing poses
- **Multi-frame Confirmation**: Validates falls over multiple frames to reduce false positives
- **Alert System**: Sound beeps and visual popups on confirmed falls
- **Clip Recording**: Saves video clips (5s before + 5s after) around fall events
- **Event Logging**: JSONL logs with timestamps and detection details

## Model Details

| Attribute | Value |
|-----------|-------|
| Model | melihuzunoglu/human-fall-detection |
| Architecture | YOLOv11 |
| Classes | Fallen, Sitting, Standing |
| Input Size | 640x640 |
| Source | Hugging Face Hub |

## Installation

```bash
# Clone or navigate to the project
cd V1

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage (Webcam)

```bash
python -m fall_detection
```

### With Video File

```bash
python -m fall_detection --source video --video-path path/to/video.mp4
```

### Custom Settings

```bash
python -m fall_detection --confidence 0.6 --cooldown-sec 30 --no-sound
```

### Headless Mode (No Display)

```bash
python -m fall_detection --no-display --save-clips
```

### All Options

```bash
python -m fall_detection --help
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | webcam | Video source (webcam/video) |
| `--video-path` | None | Path to video file |
| `--camera` | 0 | Webcam index |
| `--width` | 640 | Frame width |
| `--height` | 480 | Frame height |
| `--confidence` | 0.5 | Detection confidence threshold |
| `--fall-threshold` | 0.6 | Fall confirmation threshold |
| `--confirm-sec` | 1.5 | Confirmation duration |
| `--cooldown-sec` | 15.0 | Cooldown after alert |
| `--no-display` | False | Headless mode |
| `--save-clips` | True | Save event clips |
| `--no-sound` | False | Disable sound alerts |
| `--no-popup` | False | Disable popup alerts |
| `--device` | auto | Device (auto/cuda/cpu) |

## How It Works

```
Camera -> YOLO Detection -> State Machine -> Alert System
              |                  |               |
         [Fallen]           [NORMAL]        [Sound]
         [Sitting]          [CANDIDATE]     [Popup]
         [Standing]         [CONFIRMING]
                            [ALERTED]
                            [COOLDOWN]
```

### State Machine Flow

1. **NORMAL**: Monitoring for falls
2. **CANDIDATE**: Fall detected, validating (0.3s)
3. **CONFIRMING**: Sustained detection, confirming (1.5s)
4. **ALERTED**: Fall confirmed, alerts triggered
5. **COOLDOWN**: Waiting before next alert (15s)

## Output Structure

```
runs/
└── 20260204_123456/
    ├── events_20260204_123456.jsonl  # Event log
    └── clips/
        └── fall_20260204_123500.mp4  # Video clip
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset state to NORMAL |

## Configuration

Create a `config.yaml` file:

```yaml
model:
  confidence_threshold: 0.5
  device: "auto"

video:
  source: "webcam"
  camera_index: 0
  width: 640
  height: 480

detection:
  confirm_duration_sec: 1.5
  cooldown_duration_sec: 15.0

alert:
  enable_sound: true
  enable_popup: true

output:
  display: true
  save_clips: true
```

Then run:

```bash
python -m fall_detection -c config.yaml
```

## Project Structure

```
fall_detection/
├── __init__.py
├── __main__.py
├── cli.py                 # CLI interface
├── config/
│   ├── settings.py        # Configuration classes
│   └── default.yaml       # Default config
├── core/
│   ├── app.py             # Main application
│   └── state_machine.py   # Fall detection state machine
├── detector/
│   ├── base.py            # Base detector interface
│   └── yolo_detector.py   # YOLOv11 implementation
├── video/
│   ├── source.py          # Video input
│   └── recorder.py        # Clip recording
├── alerts/
│   ├── manager.py         # Alert coordination
│   ├── sound.py           # Sound alerts
│   └── visual.py          # Popup alerts
├── visualization/
│   └── renderer.py        # Display overlays
├── logging_/
│   └── event_logger.py    # JSONL logging
└── utils/
    └── fps_counter.py     # FPS measurement
```

## License

MIT License

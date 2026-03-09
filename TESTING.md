# Testing Fall Detection → Medtrix Backend

## Prerequisites

- Python 3.10+
- Webcam or a test video file (there's `video/falling.mp4` in the repo)

## Setup

```bash
git clone https://github.com/medtrix-AI/Fall-Detection-System.git
cd Fall-Detection-System
pip install -r requirements.txt
```

## Run with Backend Integration

### Using webcam (live detection)

```bash
python -m fall_detection \
  --source webcam \
  --backend-url http://34.62.244.146:8000 \
  --api-key dev-device-key-001 \
  --device-id 12654438-59a5-4abe-81cd-7406a38f467c \
  --location "Living Room"
```

### Using test video

```bash
python -m fall_detection \
  --source video --video-path video/falling.mp4 \
  --backend-url http://34.62.244.146:8000 \
  --api-key dev-device-key-001 \
  --device-id 12654438-59a5-4abe-81cd-7406a38f467c \
  --location "Living Room"
```

### Headless (no GUI window)

Add `--no-display` to run without the OpenCV window.

## What Happens

1. System processes video frames through YOLOv11 pose estimation
2. When a fall is detected and confirmed (~1.5s of sustained fall pose), it triggers an alert
3. The `BackendAlertHandler` sends a `POST /api/v1/edge/events` to the Medtrix backend
4. Backend creates an Alert record and broadcasts via Socket.IO
5. The mobile app receives a real-time alert notification

## Seeded Test Devices

| Device | API Key | UUID | Location |
|--------|---------|------|----------|
| Living Room Hub | `dev-device-key-001` | `12654438-59a5-4abe-81cd-7406a38f467c` | Living Room |
| Bedroom Camera | `dev-device-key-002` | `dd843ae0-5392-4f3c-9a06-e88f0eab7fa3` | Bedroom |

## Verify Alerts Reached Backend

### Quick check via API

```bash
# Login to get a token
TOKEN=$(curl -s http://34.62.244.146:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"sarah@example.com","password":"password123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['tokens']['access_token'])")

# List alerts for patient (seeded patient ID: check /patients endpoint)
curl -s http://34.62.244.146:8000/api/v1/patients/alerts \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### Manual trigger (no camera needed)

Open http://34.62.244.146:8000/tools/fall-trigger.html in a browser to send test fall events directly to the backend without running the detection system.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `webcam` | `webcam` or `video` |
| `--video-path` | — | Path to video file |
| `--camera` | `0` | Camera index |
| `--confidence` | `0.5` | YOLO detection confidence |
| `--fall-threshold` | `0.6` | Fall confirmation threshold |
| `--confirm-sec` | `1.5` | Seconds of sustained fall before alert |
| `--cooldown-sec` | `15.0` | Cooldown between alerts |
| `--backend-url` | — | Medtrix backend URL |
| `--api-key` | — | Edge device API key |
| `--device-id` | — | Edge device UUID |
| `--location` | — | Room/location label |
| `--no-display` | off | Headless mode |
| `--no-sound` | off | Disable audio alerts |
| `--no-clips` | off | Disable clip recording |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--device` | `auto` | Compute: `auto`, `cuda`, `mps`, `cpu` |

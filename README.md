# Drone Security Analyst Agent

An end-to-end autonomous drone surveillance system. Video frames are analysed by **GPT-4o-mini Vision**, reasoned over by a **stateful GPT-4o-mini security agent**, persisted in **SQLite**, and streamed live to a **dashboard** via Server-Sent Events.

<img width="1900" height="876" alt="Screenshot 2026-04-18 134819" src="https://github.com/user-attachments/assets/adfb4322-dc10-4ffe-89a5-a4204ddc2bb2" />

---

## Directory Structure

```
drone-agent/
├── api/
│   └── main.py              # FastAPI app — all endpoints + SSE simulation loop
├── agent/
│   ├── security_agent.py    # Stateful GPT-4o-mini reasoning agent (sliding window)
│   └── summarizer.py        # One-line session summary generator
├── vlm/
│   ├── model_cache.py       # Singleton OpenAI client factory
│   └── smolvlm_analyzer.py  # PNG bytes → GPT-4o-mini vision → JSON analysis
├── simulator/
│   ├── frame_generator.py   # OpenCV 640×360 synthetic frames (30 total, 3 lighting modes)
│   └── video_reader.py      # Extract frames from uploaded video files
├── database/
│   └── db.py                # SQLite WAL, 6 tables, thread-local connections
├── frontend/
│   └── index.html           # Dark tactical dashboard (SSE, Q&A, alerts, telemetry)
├── tests/
│   └── test_suite.py        # pytest test suite (40+ tests)
├── .env                     # OPENAI_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## How the Agent Achieves Temporal Reasoning

The `SecurityAgent` maintains a **sliding conversation window** of the last 20 user+model turn-pairs. Each frame's VLM analysis is appended as a new user message, so the agent has full history of every prior frame when making its current decision.

The system prompt explicitly instructs the agent to:
- Cross-reference prior frame timestamps (e.g. *"same blue truck as 01:10"*)
- Track object re-appearances across lighting changes
- Escalate when patterns accumulate over time (e.g. a person loitering across 3 consecutive frames)

This means a decision at frame 25 (night) is informed by everything seen since frame 0 (day) — without any external memory store.

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to `gpt-4o-mini`
- ~200 MB disk space (no local model weights — inference is via OpenAI API)

### Installation

```bash
# 1. Navigate to project directory
cd drone-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-proj-...
```

The server loads this automatically on startup via `python-dotenv`.

### Run the Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Open **http://localhost:8000** in a browser.

### Run Tests

```bash
pytest tests/test_suite.py -v --tb=short
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve the frontend dashboard |
| `GET`  | `/api/health` | Liveness check — returns simulation state and SSE client count |
| `GET`  | `/api/stream` | SSE stream — subscribe to real-time frame events |
| `POST` | `/api/simulate/start` | Begin the 30-frame synthetic simulation loop |
| `POST` | `/api/simulate/stop` | Interrupt a running simulation |
| `POST` | `/api/video/upload` | Upload a video file (`multipart/form-data`) and start analysis |
| `GET`  | `/api/frames` | Paginated frame list; supports `?search=<keyword>` |
| `GET`  | `/api/frames/{index}` | Single frame detail with base64 image |
| `GET`  | `/api/alerts` | List alerts; `?unacked=true` filters unacknowledged |
| `POST` | `/api/alerts/{id}/ack` | Acknowledge an alert |
| `GET`  | `/api/objects` | All detected + tracked objects |
| `GET`  | `/api/report` | Session statistics and agent decision log |
| `POST` | `/api/summarize` | Generate a one-sentence session summary |
| `POST` | `/api/qa` | Ask the agent a question using its conversation memory |

### SSE Event Types

Events are JSON objects delivered on `GET /api/stream`:

| `type` | Payload fields |
|--------|----------------|
| `connected` | `message` |
| `sim_start` | `message`, `total_frames` |
| `frame` | `frame_index`, `timestamp`, `lighting`, `severity`, `scene_description`, `objects`, `anomalies`, `tags`, `image_b64`, `agent_action`, `agent_reasoning`, `agent_confidence`, `agent_recommendation`, `telemetry`, `alert?` |
| `sim_complete` | `message`, `total_frames` |
| `sim_stopped` | `message` |

### Example: Start simulation and stream events

```bash
# Terminal 1 — start simulation
curl -X POST http://localhost:8000/api/simulate/start

# Terminal 2 — subscribe to SSE stream
curl -N http://localhost:8000/api/stream
```

### Example: Q&A after simulation

```bash
curl -X POST http://localhost:8000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "Was the blue truck seen more than once?"}'
```

---

## Synthetic Frame Scenarios

The frame generator produces 30 frames across three lighting phases:

| Phase | Frames | Lighting |
|-------|--------|----------|
| Day   | 0–12   | Bright daylight |
| Dusk  | 13–19  | Golden hour / low light |
| Night | 20–29  | Dark, NV-style |


---

## Database Schema

Six SQLite tables (WAL mode, thread-local connections):

| Table | Purpose |
|-------|---------|
| `frames` | One row per processed frame — image, VLM analysis, severity |
| `detected_objects` | Individual objects detected per frame |
| `tracked_objects` | Cross-frame object identity tracking |
| `agent_decisions` | Agent action, reasoning, confidence per frame |
| `alerts` | Raised alerts with acknowledgement support |
| `telemetry` | Simulated drone telemetry (altitude, speed, GPS, battery, heading) |

---

## Dashboard Features

- **Live feed** — frames stream in real-time with HUD overlays (timestamp, frame number, REC indicator)
- **VLM analysis** — scene description, detected objects, anomaly tags per frame
- **Agent reasoning log** — colour-coded chain-of-thought entries with action badges (MONITOR / ALERT / ESCALATE / CLEAR)
- **Telemetry strip** — altitude, speed, GPS, battery, heading with animated progress bars
- **Alerts panel** — severity stat counters (CRITICAL / HIGH / MEDIUM / LOW), flash animations, ACK button
- **Toast notifications** — non-blocking pop-up alerts for CRITICAL and HIGH events
- **Operator Q&A** — ask the agent free-form questions; answers reference specific timestamps
- **Frame search** — full-text search across scene descriptions
- **Session summary** — one-sentence summary of the entire session via `/api/summarize`
- **Video upload** — analyse your own drone footage (MP4, AVI, MOV, MKV, WebM)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision analysis | OpenAI `gpt-4o-mini` (vision) |
| Security reasoning | OpenAI `gpt-4o-mini` (chat, sliding window) |
| Session summarisation | OpenAI `gpt-4o-mini` (stateless) |
| Frame generation | OpenCV (`opencv-python-headless`) |
| API server | FastAPI + Uvicorn |
| Real-time events | Server-Sent Events (SSE) via `sse-starlette` |
| Database | SQLite (WAL mode, thread-local connections) |
| Frontend | Vanilla HTML/CSS/JS — no framework |
| Fonts | Inter + JetBrains Mono (Google Fonts) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with `gpt-4o-mini` access |


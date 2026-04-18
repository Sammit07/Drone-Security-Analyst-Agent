"""
api/main.py
FastAPI application for the Drone Security Analyst Agent.

Endpoints
---------
POST /api/simulate/start     – begin the 30-frame simulation loop (synthetic)
POST /api/simulate/stop      – stop a running simulation
POST /api/video/upload       – upload a real video file and start analysis
GET  /api/stream             – SSE stream of simulation events
GET  /api/frames             – paginated list of processed frames
GET  /api/frames/{index}     – single frame detail
GET  /api/alerts             – list of alerts (optional ?unacked=true)
POST /api/alerts/{id}/ack    – acknowledge an alert
GET  /api/objects            – all detected objects
GET  /api/report             – session statistics
POST /api/summarize          – BONUS: one-line session summary
POST /api/qa                 – BONUS: Q&A using agent conversation memory
GET  /api/health             – liveness check
GET  /                       – serve frontend index.html
"""

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.frame_generator import ALL_SPECS, render_frame_png_bytes, _fmt_time
from simulator.video_reader import extract_frames_from_bytes
from vlm.smolvlm_analyzer import analyze_frame
from agent.security_agent import get_agent, reset_agent
from agent.summarizer import generate_one_line_summary
import database.db as db

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Drone Security Analyst Agent",
    version="3.0.0",
    description="OpenAI gpt-4o-mini (vision + reasoning) end-to-end drone surveillance system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simulation state ──────────────────────────────────────────────────────────
_sim_lock = asyncio.Lock()
_sim_running = False
_sim_task: Optional[asyncio.Task] = None
_sse_queues: List[asyncio.Queue] = []

# Session events accumulator (for summarise endpoint)
_session_events: List[Dict[str, Any]] = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def _png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def _make_telemetry(frame_index: int, seconds: int) -> Dict[str, float]:
    import math
    battery = max(20.0, 100.0 - frame_index * 1.8)
    speed = 2.0 + math.sin(frame_index * 0.4) * 1.5
    lat = 37.7749 + frame_index * 0.0001
    lon = -122.4194 - frame_index * 0.0001
    heading = (frame_index * 12) % 360
    return {
        "altitude_m": 45.0,
        "speed_ms": round(speed, 2),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "battery_pct": round(battery, 1),
        "heading_deg": round(heading, 1),
    }


async def _broadcast(event_type: str, data: Dict[str, Any]) -> None:
    """Push an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, **data})
    dead: List[asyncio.Queue] = []
    for q in _sse_queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        if q in _sse_queues:
            _sse_queues.remove(q)


def _should_raise_alert(
    vlm_analysis: Dict[str, Any],
    decision: Dict[str, Any],
    frame_alert: str,
) -> Optional[Dict[str, str]]:
    """Return alert dict or None."""
    action = decision.get("action", "MONITOR")
    severity = vlm_analysis.get("severity", "low")
    anomalies = vlm_analysis.get("anomalies", [])

    if frame_alert:
        return {
            "alert_type": "SCENARIO_TRIGGER",
            "severity": severity,
            "description": frame_alert,
        }
    if action in ("ALERT", "ESCALATE"):
        desc = decision.get("recommendation", "Agent triggered alert.")
        return {
            "alert_type": action,
            "severity": severity,
            "description": desc,
        }
    if anomalies:
        return {
            "alert_type": "ANOMALY_DETECTED",
            "severity": severity,
            "description": "; ".join(str(a) for a in anomalies[:3]),
        }
    return None


# ── Shared frame-processing loop ──────────────────────────────────────────────
async def _run_frames(frame_list: List[Dict[str, Any]], total: int) -> None:
    """
    Core pipeline: accepts a list of frame dicts (from synthetic generator
    OR from video_reader) and drives them through SmolVLM → LangChain →
    SQLite → SSE.

    Each dict must have: index, timestamp_str, seconds, lighting, alert,
    png_bytes, objects  (same schema as generate_all_frames()).
    """
    global _sim_running, _session_events

    agent = get_agent()

    for fr in frame_list:
        if not _sim_running:
            await _broadcast("sim_stopped", {"message": "Simulation stopped by user."})
            return

        frame_idx = fr["index"]
        ts_str    = fr["timestamp_str"]
        lighting  = fr.get("lighting", "day")
        png_bytes = fr["png_bytes"]
        frame_alert = fr.get("alert", "")

        image_b64 = _png_to_b64(png_bytes)

        # SmolVLM
        vlm_analysis = await asyncio.to_thread(analyze_frame, png_bytes, frame_idx)

        # Persist frame
        frame_id = db.insert_frame(
            frame_index=frame_idx,
            timestamp_str=ts_str,
            seconds=fr.get("seconds", 0),
            lighting=lighting,
            image_b64=image_b64,
            vlm_analysis=vlm_analysis,
        )

        # Detected objects + tracking
        objects = vlm_analysis.get("objects", [])
        db.insert_detected_objects(frame_id, objects)
        for obj in objects:
            obj_key = f"{obj.get('type','?')}_{obj.get('color','?')}".lower()
            db.upsert_tracked_object(
                object_key=obj_key,
                timestamp_str=ts_str,
                description=f"{obj.get('color','')} {obj.get('type','')} at {obj.get('location','')}",
            )

        # Agent reasoning
        decision = await asyncio.to_thread(
            agent.process_frame, vlm_analysis, ts_str, frame_idx, lighting,
        )
        db.insert_agent_decision(frame_id, decision)

        # Telemetry
        telem = _make_telemetry(frame_idx, fr.get("seconds", 0))
        db.insert_telemetry(frame_id=frame_id, frame_index=frame_idx, **telem)

        # Alerts
        alert_info = _should_raise_alert(vlm_analysis, decision, frame_alert)
        alert_id = None
        if alert_info:
            alert_id = db.insert_alert(
                frame_id=frame_id,
                frame_index=frame_idx,
                alert_type=alert_info["alert_type"],
                severity=alert_info["severity"],
                description=alert_info["description"],
            )

        # Session summary accumulator
        _session_events.append({
            "timestamp_str": ts_str,
            "action": decision.get("action", "MONITOR"),
            "severity": vlm_analysis.get("severity", "low"),
            "scene_description": vlm_analysis.get("scene_description", ""),
            "anomalies": vlm_analysis.get("anomalies", []),
            "reasoning": decision.get("reasoning", ""),
        })

        # SSE broadcast
        await _broadcast("frame", {
            "frame_index": frame_idx,
            "timestamp": ts_str,
            "lighting": lighting,
            "image_b64": image_b64,
            "scene_description": vlm_analysis.get("scene_description", ""),
            "severity": vlm_analysis.get("severity", "low"),
            "objects": objects,
            "anomalies": vlm_analysis.get("anomalies", []),
            "tags": vlm_analysis.get("tags", []),
            "agent_action": decision.get("action", "MONITOR"),
            "agent_confidence": decision.get("confidence", 0.5),
            "agent_reasoning": decision.get("reasoning", ""),
            "agent_recommendation": decision.get("recommendation", ""),
            "alert": {
                "id": alert_id,
                "type": alert_info["alert_type"],
                "severity": alert_info["severity"],
                "description": alert_info["description"],
            } if alert_info else None,
            "telemetry": telem,
        })
        logger.info("Frame %02d/%02d processed → %s", frame_idx, total - 1,
                    decision.get("action"))
        await asyncio.sleep(0.3)

    _sim_running = False
    await _broadcast("sim_complete", {
        "message": "Analysis complete",
        "total_frames": total,
        "stats": db.get_session_stats(),
    })
    logger.info("Frame processing complete (%d frames).", total)


# ── Synthetic simulation loop ─────────────────────────────────────────────────
async def _run_simulation() -> None:
    global _sim_running, _session_events

    db.reset_db()
    reset_agent()
    _session_events = []

    frame_list = []
    for spec in ALL_SPECS:
        png_bytes = await asyncio.to_thread(render_frame_png_bytes, spec)
        frame_list.append({
            "index":         spec.index,
            "timestamp_str": _fmt_time(spec.seconds),
            "seconds":       spec.seconds,
            "lighting":      spec.lighting,
            "alert":         spec.alert_label,
            "png_bytes":     png_bytes,
            "objects":       [],
        })

    await _broadcast("sim_start", {"message": "Simulation started", "total_frames": len(frame_list)})
    logger.info("Synthetic simulation — %d frames.", len(frame_list))
    await _run_frames(frame_list, len(frame_list))


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    db.init_db()
    logger.info("Database initialised.")
    # Preload models in background thread so first request is fast
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _preload_models)


def _preload_models():
    try:
        from vlm.model_cache import _ensure_configured
        _ensure_configured()
        logger.info("OpenAI SDK configured and ready.")
    except Exception as exc:
        logger.warning("OpenAI SDK init failed (will retry on first use): %s", exc)


# ── SSE endpoint ──────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream_events(request: Request):
    """Server-Sent Events endpoint — subscribe to simulation events."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=128)
    _sse_queues.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Send keepalive immediately
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in _sse_queues:
                _sse_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Simulate endpoints ────────────────────────────────────────────────────────
@app.post("/api/simulate/start")
async def start_simulation():
    global _sim_running, _sim_task

    async with _sim_lock:
        if _sim_running:
            return JSONResponse({"status": "already_running"}, status_code=409)
        _sim_running = True
        _sim_task = asyncio.create_task(_run_simulation())

    return {"status": "started", "total_frames": 30}


@app.post("/api/simulate/stop")
async def stop_simulation():
    global _sim_running

    async with _sim_lock:
        if not _sim_running:
            return JSONResponse({"status": "not_running"}, status_code=409)
        _sim_running = False

    return {"status": "stopping"}


# ── Video upload endpoint ─────────────────────────────────────────────────────
@app.post("/api/video/upload")
async def upload_video(
    file: UploadFile = File(...),
    max_frames: int = Query(30, ge=1, le=30),
):
    """
    Upload any video file (MP4, AVI, MOV, MKV …).
    The server extracts up to *max_frames* evenly-spaced frames, resizes
    each to 640×360, then runs the full SmolVLM → LangChain → SQLite pipeline
    exactly like the synthetic simulation.

    Returns immediately with {"status": "started", "filename": ..., "total_frames": N}.
    Watch /api/stream for live frame/alert events.
    """
    global _sim_running, _sim_task, _session_events

    # Check supported extension
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    async with _sim_lock:
        if _sim_running:
            return JSONResponse({"status": "already_running"}, status_code=409)
        _sim_running = True

    # Read upload into memory
    video_bytes = await file.read()
    if not video_bytes:
        _sim_running = False
        raise HTTPException(400, detail="Uploaded file is empty.")

    # Extract frames in a thread (OpenCV is blocking)
    try:
        frame_list = await asyncio.to_thread(
            extract_frames_from_bytes, video_bytes, file.filename or "upload", max_frames
        )
    except ValueError as exc:
        _sim_running = False
        raise HTTPException(422, detail=str(exc))

    total = len(frame_list)

    async def _run_video():
        global _sim_running, _session_events
        db.reset_db()
        reset_agent()
        _session_events = []
        await _broadcast("sim_start", {
            "message": f"Video analysis started: {file.filename}",
            "total_frames": total,
            "source": "video",
            "filename": file.filename,
        })
        await _run_frames(frame_list, total)

    _sim_task = asyncio.create_task(_run_video())
    return {"status": "started", "filename": file.filename, "total_frames": total}


# ── Frame endpoints ───────────────────────────────────────────────────────────
@app.get("/api/frames")
async def list_frames(
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
):
    if search:
        frames = db.search_frames(search, limit=limit)
    else:
        frames = db.get_all_frames(limit=limit, offset=offset)

    # Strip large image_b64 from list view; include thumbnail flag
    result = []
    for f in frames:
        row = {k: v for k, v in f.items() if k != "image_b64"}
        row["has_image"] = bool(f.get("image_b64"))
        result.append(row)
    return {"frames": result, "count": len(result)}


@app.get("/api/frames/{index}")
async def get_frame(index: int):
    frame = db.get_frame(index)
    if not frame:
        raise HTTPException(404, detail=f"Frame {index} not found")
    # Parse vlm_analysis_json
    try:
        frame["vlm_analysis"] = json.loads(frame.get("vlm_analysis_json", "{}"))
    except Exception:
        frame["vlm_analysis"] = {}
    return frame


# ── Alert endpoints ───────────────────────────────────────────────────────────
@app.get("/api/alerts")
async def list_alerts(
    unacked: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
):
    alerts = db.get_alerts(unacked_only=unacked, limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/api/alerts/{alert_id}/ack")
async def acknowledge_alert(alert_id: int):
    ok = db.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, detail=f"Alert {alert_id} not found")
    return {"status": "acknowledged", "alert_id": alert_id}


# ── Objects endpoint ──────────────────────────────────────────────────────────
@app.get("/api/objects")
async def list_objects(limit: int = Query(200, ge=1, le=500)):
    objects = db.get_all_objects(limit=limit)
    tracked = db.get_tracked_objects()
    return {"objects": objects, "tracked": tracked, "count": len(objects)}


# ── Report endpoint ───────────────────────────────────────────────────────────
@app.get("/api/report")
async def get_report():
    stats = db.get_session_stats()
    decisions = db.get_agent_decisions(limit=30)
    alerts = db.get_alerts(limit=30)
    return {
        "stats": stats,
        "recent_decisions": decisions,
        "recent_alerts": alerts,
    }


# ── BONUS: Summarize ──────────────────────────────────────────────────────────
@app.post("/api/summarize")
async def summarize_session():
    if not _session_events:
        return {"summary": "No simulation data available yet. Run a simulation first."}
    summary = await asyncio.to_thread(generate_one_line_summary, _session_events)
    return {"summary": summary}


# ── BONUS: Q&A ────────────────────────────────────────────────────────────────
class QARequest(BaseModel):
    question: str


@app.post("/api/qa")
async def qa_endpoint(body: QARequest):
    if not body.question.strip():
        raise HTTPException(400, detail="question must not be empty")
    agent = get_agent()
    answer = await asyncio.to_thread(agent.answer_question, body.question)
    return {"answer": answer}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    stats = db.get_session_stats()
    return {
        "status": "ok",
        "simulation_running": _sim_running,
        "sse_clients": len(_sse_queues),
        "db_frames": stats.get("total_frames", 0),
        "db_alerts": stats.get("total_alerts", 0),
    }


# ── Frontend ──────────────────────────────────────────────────────────────────
_FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not _FRONTEND_PATH.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(_FRONTEND_PATH.read_text(encoding="utf-8"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)

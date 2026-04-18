"""
simulator/video_reader.py
Extract up to MAX_FRAMES evenly-spaced frames from any OpenCV-readable video
(MP4, AVI, MOV, MKV …) and return them in the same dict format that
frame_generator.generate_all_frames() uses, so the rest of the pipeline
(SmolVLM → LangChain → SQLite → SSE) is completely unchanged.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

TARGET_W, TARGET_H = 640, 360
MAX_FRAMES = 30
MIN_FRAMES = 1


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _add_hud(img: np.ndarray, frame_index: int,
             timestamp_str: str, source_label: str) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (TARGET_W, 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out,
                f"VIDEO  T={timestamp_str}  FRM#{frame_index:02d}  {source_label[:28]}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    cv2.putText(out, f"{TARGET_W}x{TARGET_H}",
                (4, TARGET_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 200, 0), 1)
    cx, cy = TARGET_W // 2, TARGET_H // 2
    cv2.line(out, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 1)
    cv2.circle(out, (cx, cy), 18, (0, 200, 0), 1)
    return out


def _to_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES,
                   source_label: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Open video_path, sample up to max_frames evenly-spaced frames,
    resize to 640×360, burn HUD, return list of frame dicts compatible
    with generate_all_frames() output.
    """
    max_frames = max(MIN_FRAMES, min(max_frames, MAX_FRAMES))
    label = source_label or Path(video_path).name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path!r}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        raise ValueError(f"Video has no readable frames: {video_path!r}")

    n = min(max_frames, total)
    indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]
    seen = set()
    indices = [x for x in indices if not (x in seen or seen.add(x))]

    results: List[Dict[str, Any]] = []
    for out_idx, pos in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        ts_sec = pos / fps
        bgr = _add_hud(bgr, out_idx, _fmt_time(ts_sec), label)
        results.append({
            "index": out_idx,
            "timestamp_str": _fmt_time(ts_sec),
            "seconds": round(ts_sec, 2),
            "lighting": "video",
            "alert": "",
            "png_bytes": _to_png(bgr),
            "objects": [],
        })

    cap.release()
    if not results:
        raise ValueError(f"Could not read any frames from: {video_path!r}")
    return results


def extract_frames_from_bytes(video_bytes: bytes, filename: str = "upload.mp4",
                               max_frames: int = MAX_FRAMES) -> List[Dict[str, Any]]:
    """Same as extract_frames() but accepts raw bytes from a file upload."""
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        return extract_frames(tmp_path, max_frames=max_frames, source_label=filename)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

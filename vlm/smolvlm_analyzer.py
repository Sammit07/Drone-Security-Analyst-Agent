"""
vlm/smolvlm_analyzer.py
PNG bytes → OpenAI gpt-4o-mini (vision) → structured JSON analysis.

Output schema:
  {
    "scene_description": str,
    "objects": [{"type": str, "color": str, "location": str, "confidence": float}],
    "severity": "low" | "medium" | "high" | "critical",
    "tags": [str],
    "anomalies": [str]
  }

Calls are blocking; wrap with asyncio.to_thread() at call sites.
"""

import base64
import json
import logging
import re
from typing import Any, Dict

from vlm.model_cache import get_client

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

# ── Prompt template ───────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are a drone security analyst. Analyze this aerial surveillance image and return ONLY a valid JSON object — no markdown, no extra text.

JSON schema (fill every field):
{
  "scene_description": "<one sentence describing the overall scene>",
  "objects": [
    {"type": "<vehicle|person|animal|building|unknown>", "color": "<color>", "location": "<where in frame: gate/parking/perimeter/road/building>", "confidence": <0.0-1.0>}
  ],
  "severity": "<low|medium|high|critical>",
  "tags": ["<tag1>", "<tag2>"],
  "anomalies": ["<anomaly description or empty list>"]
}

Severity guide:
  low      = normal activity, known vehicles
  medium   = unknown vehicle, loitering < 1 min
  high     = unidentified person at gate, unknown vehicle, repeated entry
  critical = armed person, forced entry, unlit vehicle at night, fire, smoke, explosion

Return ONLY the JSON object."""


# ── Fallback / default response ───────────────────────────────────────────────
_DEFAULT_ANALYSIS: Dict[str, Any] = {
    "scene_description": "Aerial drone view of a secured facility.",
    "objects": [],
    "severity": "low",
    "tags": ["facility", "drone-surveillance"],
    "anomalies": [],
}


def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Strip markdown fences if model added them
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parse failed — using default analysis")
    result = dict(_DEFAULT_ANALYSIS)

    desc_match = re.search(r'"scene_description"\s*:\s*"([^"]+)"', text)
    if desc_match:
        result["scene_description"] = desc_match.group(1)

    sev_match = re.search(r'"severity"\s*:\s*"(low|medium|high|critical)"', text)
    if sev_match:
        result["severity"] = sev_match.group(1)

    return result


def _validate_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(_DEFAULT_ANALYSIS)
    result.update(data)

    if not isinstance(result.get("objects"), list):
        result["objects"] = []
    if not isinstance(result.get("tags"), list):
        result["tags"] = []
    if not isinstance(result.get("anomalies"), list):
        result["anomalies"] = []
    if result.get("severity") not in ("low", "medium", "high", "critical"):
        result["severity"] = "low"
    if not isinstance(result.get("scene_description"), str):
        result["scene_description"] = "Surveillance frame captured."

    clean_objects = []
    for obj in result["objects"]:
        if isinstance(obj, dict):
            clean_objects.append({
                "type": str(obj.get("type", "unknown")),
                "color": str(obj.get("color", "unknown")),
                "location": str(obj.get("location", "unknown")),
                "confidence": float(obj.get("confidence", 0.8)),
            })
    result["objects"] = clean_objects
    return result


# ── Core inference function (blocking — wrap in asyncio.to_thread) ────────────
def analyze_frame(png_bytes: bytes, frame_index: int = -1) -> Dict[str, Any]:
    """
    Accepts raw PNG bytes, calls OpenAI vision API, returns validated dict.
    Blocking — must be called via asyncio.to_thread() in async contexts.
    """
    client = get_client()
    b64 = base64.b64encode(png_bytes).decode()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }],
            max_tokens=1024,
        )
        raw_text = response.choices[0].message.content
    except Exception as exc:
        logger.error("OpenAI vision API failed for frame %d: %s", frame_index, exc)
        return dict(_DEFAULT_ANALYSIS)

    logger.debug("Frame %d raw output: %s", frame_index, raw_text[:300])

    parsed = _safe_parse_json(raw_text)
    return _validate_analysis(parsed)

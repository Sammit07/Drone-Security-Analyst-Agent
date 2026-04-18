"""
agent/security_agent.py

OpenAI gpt-4o-mini-based security reasoning agent.
  - Maintains a sliding conversation window (last k=20 user+model turns).
  - process_frame()   → reasons across ALL prior frames, returns structured decision.
  - answer_question() → answers free-form questions using existing conversation memory.

All OpenAI API calls are blocking; callers must use asyncio.to_thread().
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from vlm.model_cache import get_client

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

# ── System instruction ────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = """\
You are a tactical drone security agent. You receive VLM (vision model) analyses of
surveillance frames one at a time. After each frame, you must:
1. Identify any objects/persons/vehicles.
2. Cross-reference prior frames from the conversation history (reference exact timestamps).
3. Decide on a security action: MONITOR | ALERT | ESCALATE | CLEAR.
4. Assign a confidence score 0.0–1.0.
5. Return a structured JSON decision.

Decision JSON schema:
{
  "timestamp": "<MM:SS>",
  "frame_index": <int>,
  "reasoning": "<multi-sentence chain-of-thought referencing prior frames>",
  "action": "MONITOR|ALERT|ESCALATE|CLEAR",
  "confidence": <float>,
  "priority_objects": ["<object label>"],
  "recommendation": "<one sentence>"
}

IMPORTANT: Always reference prior frame timestamps when relevant
(e.g., "same blue truck as 01:10", "person still present since 01:22").
Return ONLY the JSON object — no markdown fences."""

_WINDOW_K = 20  # keep last 20 user+model turn-pairs

# ── Agent singleton ───────────────────────────────────────────────────────────
_agent_instance: Optional["SecurityAgent"] = None


class SecurityAgent:
    """
    Stateful security reasoning agent backed by OpenAI.
    One instance per simulation session.
    """

    def __init__(self):
        self._client = get_client()
        # History: list of {"role": "user"|"assistant", "content": str}
        self._history: List[Dict[str, str]] = []
        self._frame_count = 0
        logger.info("SecurityAgent initialised.")

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _build_frame_input(
        vlm_analysis: Dict[str, Any],
        timestamp_str: str,
        frame_index: int,
        lighting: str,
    ) -> str:
        objects_str = json.dumps(vlm_analysis.get("objects", []))
        anomalies_str = json.dumps(vlm_analysis.get("anomalies", []))
        return (
            f"[T={timestamp_str} | Frame#{frame_index:02d} | {lighting.upper()}]\n"
            f"Scene: {vlm_analysis.get('scene_description', 'N/A')}\n"
            f"Objects: {objects_str}\n"
            f"Severity: {vlm_analysis.get('severity', 'low')}\n"
            f"Tags: {vlm_analysis.get('tags', [])}\n"
            f"Anomalies: {anomalies_str}"
        )

    @staticmethod
    def _safe_parse_decision(raw: str, timestamp: str, frame_index: int) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = {}
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return {
            "timestamp": data.get("timestamp", timestamp),
            "frame_index": data.get("frame_index", frame_index),
            "reasoning": data.get("reasoning", raw[:500] if raw else "No reasoning provided."),
            "action": data.get("action", "MONITOR") if data.get("action") in
                      ("MONITOR", "ALERT", "ESCALATE", "CLEAR") else "MONITOR",
            "confidence": float(data.get("confidence", 0.5)),
            "priority_objects": data.get("priority_objects", []),
            "recommendation": data.get("recommendation", "Continue monitoring."),
        }

    def _trim_history(self) -> None:
        """Keep only the last _WINDOW_K user+model turn-pairs."""
        max_messages = _WINDOW_K * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    # ── Public methods ────────────────────────────────────────────────────────
    def process_frame(
        self,
        vlm_analysis: Dict[str, Any],
        timestamp_str: str,
        frame_index: int,
        lighting: str = "day",
    ) -> Dict[str, Any]:
        """
        Feed one frame's VLM analysis into the agent.
        Returns the agent's structured decision dict.
        Blocking — wrap in asyncio.to_thread() in async code.
        """
        frame_input = self._build_frame_input(
            vlm_analysis, timestamp_str, frame_index, lighting
        )

        raw_decision = ""
        try:
            messages = (
                [{"role": "system", "content": _SYSTEM_INSTRUCTION}]
                + self._history
                + [{"role": "user", "content": frame_input}]
            )
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            raw_decision = response.choices[0].message.content
            self._history.append({"role": "user", "content": frame_input})
            self._history.append({"role": "assistant", "content": raw_decision})
            self._trim_history()
        except Exception as exc:
            logger.error("OpenAI API failed for frame %d: %s", frame_index, exc)

        self._frame_count += 1
        decision = self._safe_parse_decision(raw_decision, timestamp_str, frame_index)
        logger.info(
            "Frame %02d [%s] → action=%s confidence=%.2f",
            frame_index, timestamp_str, decision["action"], decision["confidence"],
        )
        return decision

    def answer_question(self, question: str) -> str:
        """
        Answer a free-form question using existing conversation memory.
        Returns a plain-text answer string.
        Blocking — wrap in asyncio.to_thread() in async code.
        """
        qa_input = (
            f"[OPERATOR QUESTION] {question}\n"
            "Answer based on ALL frames you have processed so far. "
            "Reference specific timestamps and objects. Be concise but thorough."
        )
        try:
            messages = (
                [{"role": "system", "content": _SYSTEM_INSTRUCTION}]
                + self._history
                + [{"role": "user", "content": qa_input}]
            )
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("answer_question failed: %s", exc)
            return "Unable to answer at this time."

    def get_memory_turns(self) -> int:
        """Return number of conversation turns currently in memory."""
        return len(self._history) // 2

    def reset(self) -> None:
        """Clear memory and frame counter (new simulation session)."""
        self._history.clear()
        self._frame_count = 0
        logger.info("SecurityAgent memory cleared.")


# ── Module-level factory ───────────────────────────────────────────────────────
def get_agent() -> SecurityAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SecurityAgent()
    return _agent_instance


def reset_agent() -> SecurityAgent:
    global _agent_instance
    _agent_instance = SecurityAgent()
    return _agent_instance

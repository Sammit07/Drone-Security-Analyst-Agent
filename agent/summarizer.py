"""
agent/summarizer.py
BONUS: generate_one_line_summary() produces a single sentence that captures
the most significant events across the full surveillance session.

Uses OpenAI gpt-4o-mini directly (no conversation history).
Blocking — wrap in asyncio.to_thread() at call sites.
"""

import logging
from typing import Any, Dict, List, Optional

from vlm.model_cache import get_client

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

_SUMMARY_PROMPT_TEMPLATE = """\
You are a security operations dispatcher. Based on the surveillance log below,
write exactly ONE sentence that summarises the most critical events of this session.
Include: the time range, the most significant threats, and the final security status.
Do NOT use bullet points or multiple sentences.

SURVEILLANCE LOG:
{log_text}

ONE-SENTENCE SUMMARY:"""


def _build_log_text(events: List[Dict[str, Any]]) -> str:
    lines = []
    for ev in events:
        ts = ev.get("timestamp_str", "??:??")
        action = ev.get("action", "MONITOR")
        sev = ev.get("severity", "low")
        desc = ev.get("scene_description", "")
        anomalies = ev.get("anomalies", [])
        anom_str = "; ".join(anomalies) if anomalies else "none"
        lines.append(
            f"[{ts}] ACTION={action} SEV={sev} | {desc} | anomalies: {anom_str}"
        )
    return "\n".join(lines) if lines else "No events recorded."


def generate_one_line_summary(
    events: List[Dict[str, Any]],
    fallback: Optional[str] = None,
) -> str:
    """
    Generate a one-sentence session summary from a list of frame event dicts.
    Blocking — wrap in asyncio.to_thread() at call sites.
    """
    if not events:
        return fallback or "No surveillance events were recorded during this session."

    log_text = _build_log_text(events)
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(log_text=log_text)

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()

        if not summary:
            raise ValueError("Empty summary returned")

        # Keep only the first sentence
        for sep in (".", "!", "?"):
            idx = summary.find(sep)
            if idx != -1:
                summary = summary[: idx + 1].strip()
                break

        return summary

    except Exception as exc:
        logger.error("generate_one_line_summary failed: %s", exc)
        return fallback or "Surveillance session completed with multiple security events detected."

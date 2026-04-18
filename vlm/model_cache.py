"""
vlm/model_cache.py
Provides a shared OpenAI Client singleton.
Reads OPENAI_API_KEY from the environment.
"""

import os
import threading
from openai import OpenAI

_client: OpenAI | None = None
_lock = threading.Lock()


def get_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on the first call (thread-safe)."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "Set OPENAI_API_KEY before starting the server."
                )
            _client = OpenAI(api_key=api_key)
    return _client


def _ensure_configured() -> None:
    """Warm up the client singleton (called at startup)."""
    get_client()

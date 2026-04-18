"""
database/db.py
SQLite database layer with WAL mode.

Tables
------
frames            – one row per processed surveillance frame
detected_objects  – objects found by SmolVLM in each frame
tracked_objects   – cross-frame object tracking
agent_decisions   – LangChain agent reasoning and actions
alerts            – raised alerts with ACK support
telemetry         – simulated drone telemetry per frame

Full-text search via search_frames() using LIKE matching on scene_description.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ── Default DB path ───────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "drone_security.db"

# ── Thread-local connection pool ──────────────────────────────────────────────
_local = threading.local()
_DB_PATH: Path = _DEFAULT_DB_PATH


def set_db_path(path: str) -> None:
    """Override the default database path (useful for tests)."""
    global _DB_PATH
    _DB_PATH = Path(path)


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if necessary."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


@contextmanager
def _cursor() -> Generator[sqlite3.Cursor, None, None]:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ── Schema ────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS frames (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_index       INTEGER NOT NULL UNIQUE,
    timestamp_str     TEXT    NOT NULL,
    seconds           INTEGER NOT NULL,
    lighting          TEXT    NOT NULL,
    image_b64         TEXT    NOT NULL,
    scene_description TEXT    DEFAULT '',
    severity          TEXT    DEFAULT 'low',
    vlm_analysis_json TEXT    DEFAULT '{}',
    created_at        TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS detected_objects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    object_type  TEXT    NOT NULL,
    color        TEXT    DEFAULT 'unknown',
    location     TEXT    DEFAULT 'unknown',
    confidence   REAL    DEFAULT 0.8,
    raw_label    TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tracked_objects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    object_key   TEXT    NOT NULL UNIQUE,
    first_seen   TEXT    NOT NULL,
    last_seen    TEXT    NOT NULL,
    appearances  INTEGER DEFAULT 1,
    description  TEXT    DEFAULT '',
    updated_at   TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id          INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    frame_index       INTEGER NOT NULL,
    timestamp_str     TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    confidence        REAL    DEFAULT 0.5,
    reasoning         TEXT    DEFAULT '',
    recommendation    TEXT    DEFAULT '',
    priority_objects  TEXT    DEFAULT '[]',
    created_at        TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER REFERENCES frames(id) ON DELETE CASCADE,
    frame_index  INTEGER NOT NULL,
    alert_type   TEXT    NOT NULL,
    severity     TEXT    NOT NULL DEFAULT 'medium',
    description  TEXT    NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS telemetry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    frame_index  INTEGER NOT NULL,
    altitude_m   REAL    DEFAULT 45.0,
    speed_ms     REAL    DEFAULT 0.0,
    latitude     REAL    DEFAULT 37.7749,
    longitude    REAL    DEFAULT -122.4194,
    battery_pct  REAL    DEFAULT 85.0,
    heading_deg  REAL    DEFAULT 0.0,
    created_at   TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frames_index      ON frames(frame_index);
CREATE INDEX IF NOT EXISTS idx_detobj_frame      ON detected_objects(frame_id);
CREATE INDEX IF NOT EXISTS idx_decisions_frame   ON agent_decisions(frame_id);
CREATE INDEX IF NOT EXISTS idx_alerts_frame      ON alerts(frame_id);
CREATE INDEX IF NOT EXISTS idx_alerts_ack        ON alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_telemetry_frame   ON telemetry(frame_id);
"""


def init_db(db_path: Optional[str] = None) -> None:
    """Create all tables if they do not exist. Call once at startup."""
    if db_path:
        set_db_path(db_path)
    with _cursor() as cur:
        cur.executescript(_DDL)
    logger.info("Database initialised at %s", _DB_PATH)


def reset_db() -> None:
    """Drop all tables and re-create. Used for fresh simulation runs."""
    drop_sql = """
    DROP TABLE IF EXISTS telemetry;
    DROP TABLE IF EXISTS alerts;
    DROP TABLE IF EXISTS agent_decisions;
    DROP TABLE IF EXISTS tracked_objects;
    DROP TABLE IF EXISTS detected_objects;
    DROP TABLE IF EXISTS frames;
    """
    with _cursor() as cur:
        cur.executescript(drop_sql)
    init_db()
    logger.info("Database reset complete.")


# ── Frames ────────────────────────────────────────────────────────────────────
def insert_frame(
    frame_index: int,
    timestamp_str: str,
    seconds: int,
    lighting: str,
    image_b64: str,
    vlm_analysis: Dict[str, Any],
) -> int:
    """Insert a frame row; returns new rowid."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT OR REPLACE INTO frames
              (frame_index, timestamp_str, seconds, lighting, image_b64,
               scene_description, severity, vlm_analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_index,
                timestamp_str,
                seconds,
                lighting,
                image_b64,
                vlm_analysis.get("scene_description", ""),
                vlm_analysis.get("severity", "low"),
                json.dumps(vlm_analysis),
            ),
        )
        return cur.lastrowid


def get_frame(frame_index: int) -> Optional[Dict[str, Any]]:
    """Fetch a single frame row by index."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM frames WHERE frame_index=?", (frame_index,))
        row = cur.fetchone()
    return dict(row) if row else None


def get_all_frames(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Fetch all frames ordered by frame_index, with pagination."""
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM frames ORDER BY frame_index LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def search_frames(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Full-text search on scene_description (LIKE).
    Returns matching frame rows ordered by frame_index.
    """
    pattern = f"%{query}%"
    with _cursor() as cur:
        cur.execute(
            """
            SELECT * FROM frames
            WHERE scene_description LIKE ?
               OR vlm_analysis_json LIKE ?
            ORDER BY frame_index
            LIMIT ?
            """,
            (pattern, pattern, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Detected objects ──────────────────────────────────────────────────────────
def insert_detected_objects(frame_id: int, objects: List[Dict[str, Any]]) -> None:
    """Bulk-insert detected objects for a frame."""
    with _cursor() as cur:
        for obj in objects:
            cur.execute(
                """
                INSERT INTO detected_objects
                  (frame_id, object_type, color, location, confidence, raw_label)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id,
                    obj.get("type", "unknown"),
                    obj.get("color", "unknown"),
                    obj.get("location", "unknown"),
                    float(obj.get("confidence", 0.8)),
                    obj.get("raw_label", ""),
                ),
            )


def get_detected_objects(frame_id: int) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM detected_objects WHERE frame_id=? ORDER BY id",
            (frame_id,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_all_objects(limit: int = 200) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            """
            SELECT do.*, f.timestamp_str, f.frame_index
            FROM detected_objects do
            JOIN frames f ON do.frame_id = f.id
            ORDER BY do.id
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Tracked objects ───────────────────────────────────────────────────────────
def upsert_tracked_object(
    object_key: str,
    timestamp_str: str,
    description: str,
) -> None:
    """Insert or update a tracked object record."""
    with _cursor() as cur:
        cur.execute(
            "SELECT id, appearances FROM tracked_objects WHERE object_key=?",
            (object_key,),
        )
        existing = cur.fetchone()
        if existing:
            cur.execute(
                """
                UPDATE tracked_objects
                SET last_seen=?, appearances=appearances+1,
                    description=?, updated_at=datetime('now')
                WHERE object_key=?
                """,
                (timestamp_str, description, object_key),
            )
        else:
            cur.execute(
                """
                INSERT INTO tracked_objects
                  (object_key, first_seen, last_seen, appearances, description)
                VALUES (?, ?, ?, 1, ?)
                """,
                (object_key, timestamp_str, timestamp_str, description),
            )


def get_tracked_objects() -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM tracked_objects ORDER BY first_seen")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Agent decisions ───────────────────────────────────────────────────────────
def insert_agent_decision(
    frame_id: int,
    decision: Dict[str, Any],
) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_decisions
              (frame_id, frame_index, timestamp_str, action, confidence,
               reasoning, recommendation, priority_objects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                decision.get("frame_index", -1),
                decision.get("timestamp", "??:??"),
                decision.get("action", "MONITOR"),
                float(decision.get("confidence", 0.5)),
                decision.get("reasoning", ""),
                decision.get("recommendation", ""),
                json.dumps(decision.get("priority_objects", [])),
            ),
        )
        return cur.lastrowid


def get_agent_decisions(limit: int = 100) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Alerts ────────────────────────────────────────────────────────────────────
def insert_alert(
    frame_id: Optional[int],
    frame_index: int,
    alert_type: str,
    severity: str,
    description: str,
) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO alerts
              (frame_id, frame_index, alert_type, severity, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (frame_id, frame_index, alert_type, severity, description),
        )
        return cur.lastrowid


def acknowledge_alert(alert_id: int) -> bool:
    with _cursor() as cur:
        cur.execute(
            "UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)
        )
        return cur.rowcount > 0


def get_alerts(
    unacked_only: bool = False,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        if unacked_only:
            cur.execute(
                "SELECT * FROM alerts WHERE acknowledged=0 ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        else:
            cur.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Telemetry ─────────────────────────────────────────────────────────────────
def insert_telemetry(
    frame_id: int,
    frame_index: int,
    altitude_m: float = 45.0,
    speed_ms: float = 0.0,
    latitude: float = 37.7749,
    longitude: float = -122.4194,
    battery_pct: float = 85.0,
    heading_deg: float = 0.0,
) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO telemetry
              (frame_id, frame_index, altitude_m, speed_ms, latitude,
               longitude, battery_pct, heading_deg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (frame_id, frame_index, altitude_m, speed_ms,
             latitude, longitude, battery_pct, heading_deg),
        )
        return cur.lastrowid


def get_telemetry(limit: int = 100) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM telemetry ORDER BY frame_index LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Stats / report ────────────────────────────────────────────────────────────
def get_session_stats() -> Dict[str, Any]:
    """Aggregate stats for the /api/report endpoint."""
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM frames")
        total_frames = cur.fetchone()["n"]

        cur.execute(
            "SELECT severity, COUNT(*) AS n FROM frames GROUP BY severity"
        )
        sev_rows = cur.fetchall()
        severity_counts = {r["severity"]: r["n"] for r in sev_rows}

        cur.execute("SELECT COUNT(*) AS n FROM alerts")
        total_alerts = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM alerts WHERE acknowledged=0")
        unack_alerts = cur.fetchone()["n"]

        cur.execute(
            "SELECT action, COUNT(*) AS n FROM agent_decisions GROUP BY action"
        )
        action_rows = cur.fetchall()
        action_counts = {r["action"]: r["n"] for r in action_rows}

        cur.execute("SELECT COUNT(*) AS n FROM detected_objects")
        total_objects = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM tracked_objects")
        total_tracked = cur.fetchone()["n"]

    return {
        "total_frames": total_frames,
        "severity_distribution": severity_counts,
        "total_alerts": total_alerts,
        "unacknowledged_alerts": unack_alerts,
        "agent_action_distribution": action_counts,
        "total_detected_objects": total_objects,
        "total_tracked_objects": total_tracked,
    }

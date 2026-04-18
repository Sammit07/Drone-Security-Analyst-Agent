"""
simulator/frame_generator.py
Renders 30 synthetic 640×360 drone-surveillance PNG frames using OpenCV.
Lighting cycles: day (frames 0-12) → dusk (frames 13-19) → night (frames 20-29).
Mandatory scenarios embedded at precise timestamps.
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

# ── Frame dimensions ────────────────────────────────────────────────────────
W, H = 640, 360

# ── Timestamp map (frame_index → seconds elapsed) ───────────────────────────
FRAME_TIMESTAMPS = [
    0,    # 00:00
    1,    # 00:01  ← blue F150 enters gate
    15,   # 00:15
    30,   # 00:30
    45,   # 00:45
    60,   # 01:00
    70,   # 01:10  ← person loitering #1
    82,   # 01:22  ← person loitering #2
    95,   # 01:35  ← person loitering #3
    110,  # 01:50
    127,  # 02:07  ← fox at perimeter
    140,  # 02:20
    155,  # 02:35
    170,  # 02:50
    185,  # 03:05
    210,  # 03:30  ← blue F150 re-enters (dusk)
    225,  # 03:45
    240,  # 04:00
    255,  # 04:15
    270,  # 04:30
    300,  # 05:00  ← white van partial plate (night)
    315,  # 05:15
    330,  # 05:30
    345,  # 05:45
    360,  # 06:00
    375,  # 06:15
    390,  # 06:30
    405,  # 06:45
    420,  # 07:00
    435,  # 07:15
]


@dataclass
class SceneObject:
    label: str
    x: int          # centre x
    y: int          # centre y
    w: int          # bounding box width
    h: int          # bounding box height
    color_bgr: Tuple[int, int, int]
    box_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    plate: str = ""


@dataclass
class FrameSpec:
    index: int
    seconds: int
    lighting: str           # "day" | "dusk" | "night"
    objects: List[SceneObject] = field(default_factory=list)
    alert_label: str = ""   # HUD banner text


# ── Lighting palette ─────────────────────────────────────────────────────────
_SKY = {
    "day":   (200, 210, 230),
    "dusk":  (40, 80, 150),
    "night": (15, 20, 30),
}
_GROUND = {
    "day":   (80, 110, 70),
    "dusk":  (40, 60, 50),
    "night": (20, 30, 20),
}
_ROAD = {
    "day":   (90, 90, 90),
    "dusk":  (55, 55, 55),
    "night": (30, 30, 30),
}
_BLDG = {
    "day":   (160, 155, 140),
    "dusk":  (90, 85, 75),
    "night": (45, 42, 38),
}


def _lighting(idx: int) -> str:
    if idx <= 12:
        return "day"
    if idx <= 19:
        return "dusk"
    return "night"


def _fmt_time(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


# ── Low-level draw helpers ────────────────────────────────────────────────────
def _draw_scene_background(img: np.ndarray, lt: str) -> None:
    """Sky gradient + ground."""
    sky = _SKY[lt]
    gnd = _GROUND[lt]
    horizon = H // 3
    for y in range(horizon):
        alpha = y / horizon
        r = int(sky[0] * (1 - alpha) + gnd[0] * alpha)
        g = int(sky[1] * (1 - alpha) + gnd[1] * alpha)
        b = int(sky[2] * (1 - alpha) + gnd[2] * alpha)
        img[y, :] = (b, g, r)
    img[horizon:, :] = gnd[::-1]   # OpenCV is BGR


def _draw_facility(img: np.ndarray, lt: str) -> None:
    """Parking lot, road, two buildings, fence line."""
    r = _ROAD[lt][::-1]
    b = _BLDG[lt][::-1]
    h = H // 3   # horizon row

    # Tarmac apron
    cv2.rectangle(img, (80, h + 20), (560, H - 40), r, -1)

    # Main building (left)
    cv2.rectangle(img, (85, h - 30), (240, h + 30), b, -1)
    cv2.rectangle(img, (85, h - 30), (240, h + 30), (40, 40, 40), 1)

    # Warehouse (right)
    cv2.rectangle(img, (370, h - 20), (555, h + 25), b, -1)
    cv2.rectangle(img, (370, h - 20), (555, h + 25), (40, 40, 40), 1)

    # Access road lane markings
    for x in range(100, 540, 40):
        cv2.line(img, (x, h + 60), (x + 20, h + 60), (200, 200, 200), 1)

    # Perimeter fence
    fence_c = (120, 120, 50) if lt == "day" else (60, 60, 25)
    cv2.line(img, (60, h - 5), (580, h - 5), fence_c, 2)
    for x in range(60, 585, 15):
        cv2.line(img, (x, h - 5), (x, h - 18), fence_c, 1)

    # Gate opening (centre of fence)
    cv2.rectangle(img, (300, h - 18), (340, h - 5), _GROUND[lt][::-1], -1)
    cv2.line(img, (300, h - 5), (300, h - 18), (0, 200, 255), 2)
    cv2.line(img, (340, h - 5), (340, h - 18), (0, 200, 255), 2)


def _draw_stars(img: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    for _ in range(60):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H // 3))
        img[y, x] = (240, 240, 240)


def _draw_vehicle(img: np.ndarray, obj: SceneObject, lt: str) -> None:
    x1, y1 = obj.x - obj.w // 2, obj.y - obj.h // 2
    x2, y2 = obj.x + obj.w // 2, obj.y + obj.h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), obj.color_bgr[::-1], -1)
    # windshield
    cv2.rectangle(img, (x1 + 4, y1 + 3), (x2 - 4, y1 + 10), (160, 200, 220), -1)
    # wheels
    for wx in [x1 + 5, x2 - 5]:
        cv2.circle(img, (wx, y2), 4, (20, 20, 20), -1)
    # bounding box
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), obj.box_color_bgr[::-1], 1)
    _label_box(img, obj.label, x1 - 2, y1 - 2, obj.box_color_bgr[::-1])
    if obj.plate:
        cv2.putText(img, obj.plate, (x1 + 2, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)


def _draw_person(img: np.ndarray, obj: SceneObject, lt: str) -> None:
    # head
    cv2.circle(img, (obj.x, obj.y - obj.h // 3), 6, obj.color_bgr[::-1], -1)
    # body
    cv2.line(img, (obj.x, obj.y - obj.h // 3 + 6),
             (obj.x, obj.y + obj.h // 4), obj.color_bgr[::-1], 3)
    # arms
    cv2.line(img, (obj.x - 8, obj.y - obj.h // 5),
             (obj.x + 8, obj.y - obj.h // 5), obj.color_bgr[::-1], 2)
    # legs
    cv2.line(img, (obj.x, obj.y + obj.h // 4),
             (obj.x - 5, obj.y + obj.h // 2), obj.color_bgr[::-1], 2)
    cv2.line(img, (obj.x, obj.y + obj.h // 4),
             (obj.x + 5, obj.y + obj.h // 2), obj.color_bgr[::-1], 2)
    x1 = obj.x - obj.w // 2
    y1 = obj.y - obj.h // 2
    x2 = obj.x + obj.w // 2
    y2 = obj.y + obj.h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), obj.box_color_bgr[::-1], 1)
    _label_box(img, obj.label, x1, y1, obj.box_color_bgr[::-1])


def _draw_animal(img: np.ndarray, obj: SceneObject) -> None:
    """Simple fox silhouette using ellipse + triangle ears."""
    cv2.ellipse(img, (obj.x, obj.y), (obj.w // 2, obj.h // 3),
                0, 0, 360, obj.color_bgr[::-1], -1)
    # head
    cv2.ellipse(img, (obj.x + obj.w // 2 - 4, obj.y - 2),
                (7, 5), 0, 0, 360, obj.color_bgr[::-1], -1)
    # tail
    pts = np.array([[obj.x - obj.w // 2, obj.y],
                    [obj.x - obj.w // 2 - 10, obj.y - 8],
                    [obj.x - obj.w // 2 - 5, obj.y + 5]], np.int32)
    cv2.fillPoly(img, [pts], obj.color_bgr[::-1])
    x1 = obj.x - obj.w // 2 - 2
    y1 = obj.y - obj.h // 2 - 2
    x2 = obj.x + obj.w // 2 + 2
    y2 = obj.y + obj.h // 2 + 2
    cv2.rectangle(img, (x1, y1), (x2, y2), obj.box_color_bgr[::-1], 1)
    _label_box(img, obj.label, x1, y1, obj.box_color_bgr[::-1])


def _label_box(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


def _draw_hud(img: np.ndarray, spec: FrameSpec) -> None:
    """Heads-up display overlay: timestamp, frame #, coords, severity band."""
    overlay = img.copy()
    # Top-left panel
    cv2.rectangle(overlay, (0, 0), (200, 18), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    ts = _fmt_time(spec.seconds)
    cv2.putText(img, f"DRONE-CAM  T={ts}  FRM#{spec.index:02d}",
                (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

    # Bottom-left: coordinates
    lat = 37.7749 + spec.index * 0.0001
    lon = -122.4194 - spec.index * 0.0001
    cv2.putText(img, f"LAT {lat:.4f}  LON {lon:.4f}  ALT 45m",
                (4, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 0), 1)

    # Top-right: lighting mode
    lt_col = {"day": (0, 220, 255), "dusk": (0, 140, 255), "night": (150, 80, 255)}
    cv2.putText(img, spec.lighting.upper(),
                (W - 55, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lt_col[spec.lighting], 1)

    # Alert banner
    if spec.alert_label:
        cv2.rectangle(img, (0, H - 30), (W, H - 18), (0, 0, 180), -1)
        cv2.putText(img, f"!! {spec.alert_label} !!",
                    (W // 2 - 100, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # Reticle crosshair (centre)
    cx, cy = W // 2, H // 2
    cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
    cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)
    cv2.circle(img, (cx, cy), 20, (0, 200, 0), 1)


# ── Scene object definitions for each frame ──────────────────────────────────
_BLUE_F150 = SceneObject("F150[BLU]", 320, 205, 46, 22,
                          (0, 80, 200), (0, 220, 255))
_BLUE_F150_INSIDE = SceneObject("F150[BLU]", 280, 215, 46, 22,
                                 (0, 80, 200), (0, 220, 255))
_BLUE_F150_PARKED = SceneObject("F150[BLU]", 200, 220, 46, 22,
                                 (0, 80, 200), (0, 220, 255))
_BLUE_F150_EXIT = SceneObject("F150[BLU]", 350, 200, 46, 22,
                               (0, 80, 200), (0, 220, 255))
_WHITE_VAN = SceneObject("VAN[WHT] LP:7??-XR4", 430, 222, 54, 24,
                          (230, 230, 230), (0, 60, 255), plate="7??-XR4")
_PERSON_GATE = SceneObject("PERSON", 318, 188, 16, 32,
                            (210, 170, 120), (0, 60, 255))
_PERSON_GATE2 = SceneObject("PERSON", 322, 186, 16, 32,
                             (210, 170, 120), (0, 60, 255))
_PERSON_GATE3 = SceneObject("PERSON", 315, 190, 16, 32,
                             (210, 170, 120), (0, 60, 255))
_FOX = SceneObject("ANIMAL/FOX", 95, 175, 28, 14, (30, 110, 200), (0, 255, 128))
_SEDAN_A = SceneObject("SEDAN[GRY]", 160, 218, 40, 18, (120, 120, 120), (0, 200, 100))
_SEDAN_B = SceneObject("SEDAN[RED]", 480, 225, 40, 18, (30, 30, 180), (0, 200, 100))
_PATROL_TRUCK = SceneObject("TRUCK[BLK]", 250, 225, 48, 22,
                             (40, 40, 40), (0, 200, 100))


def _build_specs() -> List[FrameSpec]:
    specs: List[FrameSpec] = []
    for idx, secs in enumerate(FRAME_TIMESTAMPS):
        lt = _lighting(idx)
        objs: List[SceneObject] = []
        alert = ""

        if idx == 1:    # 00:01 blue F150 entering gate
            objs = [_BLUE_F150]
            alert = "VEHICLE ENTERING - BLUE PICKUP"
        elif idx == 2:
            objs = [_BLUE_F150_INSIDE]
        elif idx == 3:
            objs = [_BLUE_F150_PARKED, _SEDAN_A]
        elif idx == 4:
            objs = [_SEDAN_A, _SEDAN_B]
        elif idx == 5:
            objs = [_SEDAN_A, _SEDAN_B]
        elif idx == 6:  # 01:10 person loitering #1
            objs = [_PERSON_GATE, _SEDAN_A]
            alert = "PERSON LOITERING AT GATE"
        elif idx == 7:  # 01:22 person loitering #2
            objs = [_PERSON_GATE2, _SEDAN_A]
            alert = "PERSON LOITERING AT GATE"
        elif idx == 8:  # 01:35 person loitering #3
            objs = [_PERSON_GATE3]
            alert = "PERSON LOITERING - EXTENDED DWELL"
        elif idx == 9:
            objs = [_SEDAN_B]
        elif idx == 10:  # 02:07 fox at perimeter (false positive)
            objs = [_FOX]
            alert = "ANIMAL DETECTED AT PERIMETER"
        elif idx == 11:
            objs = [_SEDAN_A]
        elif idx == 12:
            objs = [_SEDAN_A, _SEDAN_B]
        elif idx == 13:
            objs = [_SEDAN_B]
        elif idx == 14:
            objs = [_SEDAN_A, _PATROL_TRUCK]
        elif idx == 15:  # 03:30 blue F150 re-enters (dusk)
            objs = [_BLUE_F150_EXIT]
            alert = "BLUE F150 RETURNING - SECOND ENTRY"
        elif idx == 16:
            objs = [_BLUE_F150_INSIDE]
        elif idx == 17:
            objs = [_BLUE_F150_PARKED, _PATROL_TRUCK]
        elif idx == 18:
            objs = [_PATROL_TRUCK]
        elif idx == 19:
            objs = [_PATROL_TRUCK, _SEDAN_B]
        elif idx == 20:  # 05:00 white van partial plate (night)
            objs = [_WHITE_VAN]
            alert = "UNIDENTIFIED VAN - PARTIAL PLATE"
        elif idx == 21:
            objs = [_WHITE_VAN, _PATROL_TRUCK]
        elif idx == 22:
            objs = [_PATROL_TRUCK]
        elif idx == 23:
            objs = [_SEDAN_B]
        elif idx == 24:
            objs = [_SEDAN_B, _PATROL_TRUCK]
        elif idx == 25:
            objs = [_PATROL_TRUCK]
        elif idx >= 26:
            objs = []

        specs.append(FrameSpec(index=idx, seconds=secs,
                               lighting=lt, objects=objs, alert_label=alert))
    return specs


# ── Main render function ─────────────────────────────────────────────────────
def render_frame(spec: FrameSpec) -> np.ndarray:
    """Return a 640×360 BGR numpy array for the given FrameSpec."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    _draw_scene_background(img, spec.lighting)
    if spec.lighting == "night":
        _draw_stars(img)
    _draw_facility(img, spec.lighting)

    for obj in spec.objects:
        if "PERSON" in obj.label:
            _draw_person(img, obj, spec.lighting)
        elif "ANIMAL" in obj.label or "FOX" in obj.label:
            _draw_animal(img, obj)
        else:
            _draw_vehicle(img, obj, spec.lighting)

    _draw_hud(img, spec)

    # Night-mode luminance reduction
    if spec.lighting == "night":
        img = (img.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
        # Add sensor noise grain
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif spec.lighting == "dusk":
        img = (img.astype(np.float32) * 0.75).clip(0, 255).astype(np.uint8)

    return img


def render_frame_png_bytes(spec: FrameSpec) -> bytes:
    """Return PNG-encoded bytes for the given FrameSpec."""
    img = render_frame(spec)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for frame {spec.index}")
    return buf.tobytes()


# ── Public API: generate all 30 frames ───────────────────────────────────────
ALL_SPECS: List[FrameSpec] = _build_specs()


def generate_all_frames() -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      { index, timestamp_str, seconds, lighting, alert, png_bytes }
    """
    results = []
    for spec in ALL_SPECS:
        png = render_frame_png_bytes(spec)
        results.append({
            "index": spec.index,
            "timestamp_str": _fmt_time(spec.seconds),
            "seconds": spec.seconds,
            "lighting": spec.lighting,
            "alert": spec.alert_label,
            "png_bytes": png,
            "objects": [o.label for o in spec.objects],
        })
    return results


if __name__ == "__main__":
    import os, base64

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output_frames")
    os.makedirs(out_dir, exist_ok=True)
    for fr in generate_all_frames():
        path = os.path.join(out_dir, f"frame_{fr['index']:02d}_{fr['timestamp_str'].replace(':','')}.png")
        with open(path, "wb") as f:
            f.write(fr["png_bytes"])
    print(f"Saved 30 frames to {out_dir}")

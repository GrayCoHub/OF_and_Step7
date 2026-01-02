# optical_flow_HO_refactored_chatgpt5_P2.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import sys
import argparse
import requests
from requests.auth import HTTPDigestAuth
import os
import uuid
import glob

'''
To run the code from PS and set the Watcher.  Remains off by default for single video processing. 

    $env:STD_PATHS_WATCHER = "0"
    python .\optical_flow_HO_refactored_chatgpt5.py
    
   What “soft” vs “hard” means (Pass-1 gating)

    soft
    Definition: Don’t apply the spatial mask when counting motion.
    Effect: active = count(motion_mask) over the entire frame.
    Implementation: No bitwise_and with the keep mask before counting.

    hard
    Definition: Apply the spatial mask before counting motion.
    Effect: Zero-out all pixels outside the allowed regions (per water/ROI rules) and then count.
    Implementation: motion_mask = motion_mask & keep_mask and then active = count(motion_mask).
    
    What “soft” vs “hard” means (Pass-2 gating)
    Pass-1 / Pass-2 “soft” = count motion everywhere (ignore the keep mask).
    
    




# ps  Option A — variable (recommended)

$ov = '{"MASK_GATING_PASS1":"hard","MOTION_THRESHOLD":2.4,"MIN_MOTION_PIXELS":170,"MIN_SEGMENT_LENGTH":8,"MERGE_GAP_FRAMES":16}'
python .\optical_flow_HO_refactored_chatgpt5.py --overrides $ov


# Use Multiline here-string (easiest) when large number of variables w/in ' {5 vars }'
# best yet? 

$ov = '{
  "MASK_GATING_PASS1": "hard",
  "MASK_GATING_PASS2": "soft",

  "MOTION_THRESHOLD": 2.0,
  "MIN_MOTION_PIXELS": 100,

  "MOTION_THRESHOLD_P2": 1.6,
  "MIN_MOTION_PIXELS_P2": 70,
  "P2_PAD_FRAMES": 6,

  "MERGE_GAP_FRAMES": 24,
  "MORPH_OPEN_ITERS": 0,
  "MORPH_CLOSE_ITERS": 1
}'

python .optical_flow_HO_refactored_chatgpt5_P2.py --overrides $ov


The overrides.json File Location: C:\AxisRecordings\EagleDetection\outputs\overrides_file

'''

# 0) Force watcher OFF for OF runs unless the caller deliberately set it earlier
os.environ["STD_PATHS_WATCHER"] = os.environ.get("STD_PATHS_WATCHER", "0")

# 1) Output paths (std_paths_of if available; otherwise local fallback)
#    This script is sometimes run outside the EagleDetection repo (e.g., OF_vs_Step7 project),
#    so coreModules_Eagle may not be importable. In that case we fall back to a tiny local
#    path helper that creates: <output_root>/<MODULE>/<run_name>/{logs,frames,videos}

def _now_stamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class _LocalStdPaths:
    def __init__(self):
        self._output_root = None

    def set_output_root(self, root: str):
        self._output_root = str(root)

    def get_paths(self, video_path: str, module_name: str):
        from pathlib import Path
        vp = Path(video_path)
        stem = vp.stem
        run_name = f"{stem}_{_now_stamp()}_{uuid.uuid4().hex[:4]}"
        out_root = Path(self._output_root) if self._output_root else Path.cwd() / "outputs"
        run_dir = out_root / module_name / run_name
        logs = run_dir / "logs"
        frames = run_dir / "frames"
        videos = run_dir / "videos"
        for d in (logs, frames, videos):
            d.mkdir(parents=True, exist_ok=True)
        return {
            "OUT_DIR": str(run_dir),
            "LOG_DIR": str(logs),
            "IMAGE_DIR": str(frames),  # historical key used in code
            "VIDEO_DIR": str(videos),
            "RUN_NAME": run_name,
        }

try:
    from coreModules_Eagle import std_paths_of as sp  # type: ignore
except Exception:
    sp = _LocalStdPaths()

# 2) Optional outputs root via env (pick one name and stick to it)
_OF_ENV_ROOT = os.getenv("OF_OUTPUT_ROOT")  # optional override for outputs root
if _OF_ENV_ROOT:
    sp.set_output_root(_OF_ENV_ROOT)

# 2b) Hard-coded output root for OF vs Step7 comparisons
#     Forces all OF_HO outputs under: C:\Axis_code_projects\OF_vs_Step7\outputs\Step7_full\<run>\{logs,frames,videos}
_HARDCODED_OUTPUT_ROOT = r"C:\Axis_code_projects\OF_vs_Step7\outputs"
sp.set_output_root(_HARDCODED_OUTPUT_ROOT)

MODULE_NAME = "Step7_full"  # outputs aligned under Step7_full
SCRIPT_VERSION = "2.0"

# ======================================================================================
# INPUT CONFIG (HARD-CODED, FORGET-PROOF)
# VIDEO_ROOT IS ALWAYS A FOLDER.
# ======================================================================================

VIDEO_ROOT = Path(r"C:\AxisRecordings\Optical_Flow\videos")  # <-- always a folder

# Choose one:
INPUT_MODE = "single"   # "single" or "batch"

# Used only if INPUT_MODE == "single"
SINGLE_CLIP_NAME = "big_bird_R2L.mkv"  # file must exist inside VIDEO_ROOT

# Extensions searched in batch mode (and validated in single mode)
VIDEO_EXTS = (".mkv",)  # add ".mp4" if needed

# Basic validation (fail fast; saves future-you from confusion)
if not VIDEO_ROOT.exists() or not VIDEO_ROOT.is_dir():
    raise SystemExit(f"VIDEO_ROOT must be an existing folder:\n  {VIDEO_ROOT}")

if INPUT_MODE not in ("single", "batch"):
    raise SystemExit(f"INPUT_MODE must be 'single' or 'batch' (got {INPUT_MODE!r})")

if INPUT_MODE == "single":
    _single_path = VIDEO_ROOT / SINGLE_CLIP_NAME
    if not _single_path.exists():
        raise SystemExit(f"Single clip not found:\n  {_single_path}")
    if _single_path.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(
            f"Single clip extension not allowed: {_single_path.suffix}\n"
            f"Allowed: {VIDEO_EXTS}"
        )

# NOTE:
# Do NOT call sp.set_clip_override(...) here.
# That creates hidden behavior and makes it harder to reason about input selection.
# Input selection should be explicit: VIDEO_ROOT + INPUT_MODE (+ SINGLE_CLIP_NAME).

PREVIEW_DEFAULT_SOURCE = "clip"

# Used w/ the choice [2]
PREVIEW_REFERENCE_DIR = Path(r"output/eagleEngine_mass_prod/frame_metadata")
PREVIEW_REFERENCE_PRESET = "20251014_135916_2154_20251014_224421"

DOWNLOADS_DIR_DEFAULT = Path(r"C:\Users\prior\Downloads")
DOWNLOADS_DIR = Path(r"C:\Users\prior\Downloads")
DOWNLOAD_SNAPSHOT_GLOB = "snapshot_*.jpg"

PROFILE_BASE_W = 1920
PROFILE_BASE_H = 1080

STATUS_EVERY = 25
AXIS_IP_DEFAULT = "192.168.1.146"
AXIS_CREDS_PATH_DEFAULT = r"C:\All_api_keys\axis_q6155\axis_creds.json"



# -----------------------------------------------------------------------------
# ROI - Region of Interest
# GLOBAL ROIs "QUICK TOGGLES" (used by PREVIEW overlay only)
# -----------------------------------------------------------------------------
# NOTE: The actual processing mask comes from CAMERA_PRESETS[preset]['rois'] inside
# build_keep_mask(). These global ROIs exist only for the PREVIEW overlay (draw_preview_caption)
# to let you visualize boxes without changing preset data.
# In other words:
#   - Preview overlay uses globals: ROIS / ROIS_NAMES
#   - Runtime masking uses preset-level ROIs from CAMERA_PRESETS[preset]['rois']
# If you want the preview to mirror the preset exactly, leave these globals empty and rely
# on the preset definitions (which are always used at runtime).

# --- Water Mask (easy toggles) ---
WATER_MASK_ENABLED   = True           # ← master on/off
WATER_MASK_MODE      = "exclude"      # "exclude" or "include"
WATER_MASK_DILATE_PX = 15
WATER_SHIFT_X        = 0
WATER_SHIFT_Y_TOP    = -185
WATER_SHIFT_Y_BOTTOM = 5
# New: horizontal scaling controls (preview + runtime)
WATER_SCALE_X        = 1.0       # 1.0 = no change; 0.75 = shrink to 75% width
WATER_ANCHOR_X       = "center"  # "left" | "center" | "right"



# --- Wind (ROI) filters: define once, pick any combo ---
ROI_FILTER_ENABLED = False           # toggle ON/OFF
ROI_FILTER_MODE    = "exclude"      # "exclude" or "include"

# Define the four named rectangles (x1, y1, x2, y2) [ Depecated - use Camera_Presets]
'''
bush_left     = (0, 775, 375, 1079)
grass_bottom  = (0, 900, 1919, 1079)
bush_right    = (1450, 725, 1919, 1079)
tree_stand    = (1250, 0, 1919, 725)

# Choose the ones you want by listing them here:
ROIS = [
    # bush_left,
    # grass_bottom,
    # bush_right,
    # tree_stand,
]

# (Optional) names to show in captions/logs, same order as ROIS
ROIS_NAMES = [
    "bush_left",
    "grass_bottom",
    "bush_right",
    "tree_stand",
]
'''

# Water mask profiles: (x, y, z) where y is top edge, z is bottom edge
HOME_PROFILE_ABS = [
    (0, 700, 1090), (100, 690, 1080), (200, 650, 1075), (300, 670, 1070), (400, 655, 1060),
    (500, 640, 1060), (600, 625, 1055), (700, 610, 1050), (800, 600, 1040), (900, 590, 1040),
    (1000, 580, 1020), (1100, 570, 1010), (1200, 560, 1000), (1300, 550, 990), (1400, 540, 980),
    (1450, 530, 970), (1600, 520, 935), (1700, 510, 920), (1800, 490, 895), (1920, 480, 900),
]

HARBOR_PROFILE_ABS = [
    (0, 900, 1090), (100, 895, 1080), (200, 885, 1075), (300, 875, 1070), (400, 865, 1065),
    (500, 850, 1060), (600, 825, 1055), (700, 810, 1050), (800, 800, 1040), (900, 790, 1040),
    (1000, 780, 1020), (1100, 770, 1010), (1200, 760, 1000), (1300, 750, 990), (1400, 740, 980),
    (1450, 730, 970), (1600, 720, 935), (1700, 710, 920), (1800, 690, 895), (1920, 680, 900),
]


# ============= Camera Presets =======================================================

# -----------------------------------------------------------------------------
# CAMERA PRESETS - Scene mask presets that are used for both the Previews and Runtime

# Only for masking, caption overlays, and camera geometry, not motion logic
# -----------------------------------------------------------------------------
# 
CAMERA_PRESETS = {
    "Home": {
        "ptz": {"pan_deg": 0.0, "tilt_deg": 0.0, "zoom": 1.0},
        "water_profile": HOME_PROFILE_ABS,
        "rois": [
            #(1250, 0, 1919, 725),   # tree   (x1<=x2, y1<=y2)
            # (0, 900, 1919, 1079),   # grass
            # (1450, 725, 1919, 1079),# right_bush (clip to 1079)
            # (0, 775, 375, 1079),    # left_bush (ordered)
        ],
        "rois_names": ["tree", "grass", "right_bush", "left_bush"],
    },
    "Harbor": {
        "ptz": {"pan_deg": 0.0, "tilt_deg": 8.0, "zoom": 240.0},
        "water_profile": HARBOR_PROFILE_ABS,
        "rois": [
            (200, 900, 400, 1000),    # harbor_left
            (1200, 100, 1800, 600),   # harbor_right
            (0, 900, 1919, 1079),     # grass
        ],
        "rois_names": ["harbor_left", "harbor_right", "grass"],
    },
}

# ===================  End Camera Presets ============================================


    
def _water_polygon(preset_data: dict, config: Config):
    """
    Build a single Nx2 int32 polygon from the water_profile and the Config shifts/scales.
    Geometry is done in the profile's native 1920x1080 coordinates:
      - X translation: WATER_SHIFT_X
      - Y independent shifts: WATER_SHIFT_Y_TOP / WATER_SHIFT_Y_BOTTOM
      - X scaling: WATER_SCALE_X anchored at WATER_ANCHOR_X ("left"|"center"|"right")
    Returns (poly, npts) or (None, 0).
    """

    profile = preset_data.get("water_profile") or []
    if not profile:
        return (None, 0)  # ← ensure tuple

    dx  = int(getattr(config, "WATER_SHIFT_X", 0))
    dyt = int(getattr(config, "WATER_SHIFT_Y_TOP", 0))
    dyb = int(getattr(config, "WATER_SHIFT_Y_BOTTOM", 0))
    scale_x  = float(getattr(config, "WATER_SCALE_X", 1.0))
    anchor_s = str(getattr(config, "WATER_ANCHOR_X", "center")).lower()

    if anchor_s == "left":
        ax = 0.0
    elif anchor_s == "right":
        ax = 1920.0
    else:
        ax = 1920.0 / 2.0  # center

    top, bot = [], []
    for (x, y, z) in profile:
        xt = float(x + dx)
        yt = int(y + dyt)
        zb = int(z + dyb)

        xs = ax + (xt - ax) * scale_x

        xs_i = int(min(1919, max(0, round(xs))))
        yt   = int(min(2000, max(-2000, yt)))
        zb   = int(min(2000, max(-2000, zb)))

        top.append((xs_i, yt))
        bot.append((xs_i, zb))

    pts = top + bot[::-1]
    if len(pts) < 3:
        return (None, 0)  # ← ensure tuple

    return (np.asarray(pts, dtype=np.int32), len(pts))  # ← always a tuple



@dataclass
class Config:
    MOTION_THRESHOLD: float = 2.0      # aka FLOW_MAG_MIN
    MIN_MOTION_PIXELS: int = 100
    # --- Pass-2 recovery (optional; used to add MD frames inside/around P1 segments) ---
    MOTION_THRESHOLD_P2: float = 1.6
    MIN_MOTION_PIXELS_P2: int = 70
    P2_PAD_FRAMES: int = 6
    MIN_SEGMENT_LENGTH: int = 6
    PADDING_FRAMES: int = 20
    MERGE_GAP_FRAMES: int = 10
    MASK_GATING_PASS1: str = "soft"
    MASK_GATING_PASS2: str = "hard"
    PRIMARY_DIRECTION: str = "any"
    PRIMARY_MIN_FRAMES: int = 4   # 8
    PRIMARY_MIN_SPEED: float = 2.0
    W_IOU: float = 4.0
    W_DX: float = 2.0
    W_AREA: float = 0.2
    W_FLOW: float = 0.6
    W_ROI_EXCL: float = 12.0
    W_WATER: float = 6.0
    MAX_JUMP_PX: int = 60
    ROI_EXCL_HARDVETO: float = 0.70
    USE_MORPHOLOGY: bool = True
    MORPH_KERNEL_SZ: int = 3
    MORPH_OPEN_ITERS: int = 0
    MORPH_CLOSE_ITERS: int = 2
    BOX_OVERLAY: bool = True
    BOX_MIN_PIXELS: int = 100
    MAX_FILES: int = 999999
    RECURSIVE: bool = True
    SAVE_FRAME_JPEGS: bool = True
    SAVE_PER_SEGMENT_VIDEO: bool = True
    SAVE_COMPILED_VIDEO: bool = True
    BOX_THICKNESS: int = 2
    BOX_DASH_LEN: int = 12
    BOX_GAP_LEN: int = 8
    SAVE_TRACKED_VIDEO: bool = True
    LOG_TRACKS: bool = True
    CANDIDATE_MIN_AREA: int = 20
    Y_FILTER_ENABLED: bool = False
    Y_FILTER_MODE: str = "inside"
    Y_RANGE: Tuple[int, int] = (200, 600)
    X_FILTER_ENABLED: bool = False
    X_FILTER_MODE: str = "inside"
    X_RANGES: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 0), (1800, 1919)])
    ROI_FILTER_ENABLED: bool = True
    ROI_FILTER_MODE: str = "exclude"
    WATER_MASK_ENABLED: bool = True
    WATER_MASK_MODE: str = "exclude"
    WATER_MASK_DILATE_PX: int = 2
    WATER_SHIFT_X: int = 0
    WATER_SHIFT_Y_TOP: int = 0
    WATER_SHIFT_Y_BOTTOM: int = 0
    CAPTION_MARGIN: int = 12
    CAPTION_FONTSCALE: float = 0.6
    CAPTION_THICKNESS: int = 1

    # --- Terminal diagnostics / segment quality ---
    # Density = (MD frames inside segment span) / (span length).
    # This helps catch over-merging: e.g., 2 MD frames bridged across a huge gap can still form a "segment".
    LOW_DENSITY_WARN: float = 0.20  # warn if segment density falls below this (20%)


@dataclass
class State:
    override_info: Dict[str, Any] = None
    camera_snapshot: Dict[str, Any] = None
    requested_preset: Optional[str] = None

    def __post_init__(self):
        self.override_info = {
            "source": None,
            "preset_name": "None",
            "values": {},
            "mode": "preset",
            "enabled_count": 0,
            "applied_keys": [],
        }
        self.camera_snapshot = {}


ALLOWED_OVERRIDES = {
    "MOTION_THRESHOLD": float,       # aka FLOW_MAG_MIN
    "MIN_MOTION_PIXELS": int,
    "MOTION_THRESHOLD_P2": float,
    "MIN_MOTION_PIXELS_P2": int,
    "P2_PAD_FRAMES": int,
    "MIN_SEGMENT_LENGTH": int,
    "PADDING_FRAMES": int,
    "MERGE_GAP_FRAMES": int,
    "MASK_GATING_PASS1": str,
    "MASK_GATING_PASS2": str,
    "PRIMARY_DIRECTION": str,
    "PRIMARY_MIN_FRAMES": int,
    "PRIMARY_MIN_SPEED": float,
    "W_IOU": float,
    "W_DX": float,
    "W_AREA": float,
    "W_FLOW": float,
    "W_ROI_EXCL": float,
    "W_WATER": float,
    "MAX_JUMP_PX": int,
    "ROI_EXCL_HARDVETO": float,
    "USE_MORPHOLOGY": bool,
    "MORPH_KERNEL_SZ": int,
    "MORPH_OPEN_ITERS": int,      # Removes small noise blobs (Erosion followed by dilation)
    "MORPH_CLOSE_ITERS": int,      #  Fills small holes and gaps (Dilation followed by erosion)
    "BOX_OVERLAY": bool,
    "BOX_MIN_PIXELS": int,  
    "WATER_MASK_ENABLED": bool,
    "ROI_FILTER_ENABLED": bool,    
}
'''
MORPH_CLOSE_ITERS: 2
Closing applied twice
Fills small holes in motion regions - preserve motion but fill gaps caused by texture or lighting

Helps unify fragmented motion blobs (e.g. wings, shadows, snow gaps)
'''

# === Hard blacklist: JSON cannot change any mask/shift fields ===
DISALLOWED_MASK_KEYS = {    
    "WATER_MASK_MODE",
    "WATER_MASK_DILATE_PX",
    "WATER_SHIFT_X",
    "WATER_SHIFT_Y_TOP",
    "WATER_SHIFT_Y_BOTTOM",
    "WATER_SCALE_X",           # ← add new
    "WATER_ANCHOR_X",          # ← add new    
    "ROI_FILTER_MODE",
    "X_FILTER_ENABLED", "X_FILTER_MODE", "X_RANGES",
    "Y_FILTER_ENABLED", "Y_FILTER_MODE", "Y_RANGE",
    "ROIS", "ROIS_NAMES",    
}


OVERRIDES_FILENAME = "of_overrides.json"
# Canonical, user-edited overrides file path (stable; exists before runs)
OVERRIDES_CANON_PATH = Path(r"C:\AxisRecordings\EagleDetection\outputs\overrides_file\of_overrides.json")
ENABLE_OVERRIDES = True


def apply_overrides(
    config: Config,
    override_dict: Dict[str, Any],
    source: str,
    preset_name: str = "custom",
) -> Dict[str, Any]:
    """Apply overrides from a dictionary to the config object, supporting sparse and template modes."""
    info = {
        "source": source,
        "preset_name": preset_name,
        "values": {},
        "mode": "sparse",
        "enabled_count": 0,
        "applied_keys": [],
    }

    if not override_dict:
        print(f" Overrides: empty from {source}, preset: {preset_name}")
        return info

    info["source"] = source
    flat_overrides: Dict[str, Any] = {}

    if override_dict.get("__mode") == "template":
        info["mode"] = "template"
        knobs = override_dict.get("knobs", {})
        for k, spec in knobs.items():
            if isinstance(spec, dict) and spec.get("use"):
                flat_overrides[k] = spec.get("value")
        tracking = override_dict.get("tracking", {})
        for section in ["mask_gating", "primary", "weights", "guards"]:
            for k, spec in tracking.get(section, {}).items():
                if isinstance(spec, dict) and spec.get("use"):
                    flat_overrides[k] = spec.get("value")
    else:
        flat_overrides = override_dict
        info["mode"] = "sparse"

    info["values"] = flat_overrides
    info["enabled_count"] = len(flat_overrides)
    info["applied_keys"] = list(flat_overrides.keys())

    for key, value in flat_overrides.items():
        # hard-block mask/shift edits from JSON
        if key in DISALLOWED_MASK_KEYS:
            print(f"⚠️  Ignoring override for '{key}' (mask/shift keys are not settable via JSON).")
            continue

        if key in ALLOWED_OVERRIDES:
            try:
                if key in ("MASK_GATING_PASS1", "MASK_GATING_PASS2") and value not in ("soft", "hard"):
                    print(f" Invalid value for {key}: {value}; must be 'soft' or 'hard'")
                    continue
                setattr(config, key, ALLOWED_OVERRIDES[key](value))
            except (ValueError, TypeError) as e:
                print(f" Invalid override for {key}: {value} ({e})")
        else:
            print(f" Unknown override key: {key}")

    if flat_overrides:
        print(f"[OVERRIDES] preset: {preset_name} ({info['enabled_count']} key(s) from {source}: {info['applied_keys']})")
    else:
        print(f" Overrides: empty from {source}, preset: {preset_name}")

    return info


def apply_overrides_from_file(session_root: Path, config: Config, preset_name: str = "custom") -> Dict[str, Any]:
    """Load and apply overrides from of_overrides.json (sparse or template mode)."""
    info = {"source": None, "preset_name": preset_name, "values": {}, "mode": "preset", "enabled_count": 0, "applied_keys": []}
    path = session_root / OVERRIDES_FILENAME
    if not path.exists():
        print(f" Overrides: none found at {path}")
        return info

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception as e:
        print(f" Overrides error: failed to load {path}: {e}")
        return info

    if not isinstance(payload, dict):
        print(f" Overrides: invalid JSON structure in {path}")
        return info

    info["source"] = str(path)
    flat_overrides: Dict[str, Any] = {}
    if payload.get("__mode") == "template":
        # Flatten template mode (non-mask only)
        knobs = payload.get("knobs", {})
        for k, spec in knobs.items():
            if isinstance(spec, dict) and spec.get("use"):
                flat_overrides[k] = spec.get("value")
        info["mode"] = "template"
    else:
        # Sparse mode
        flat_overrides = payload
        info["mode"] = "sparse"

    info["values"] = flat_overrides
    info["enabled_count"] = len(flat_overrides)
    info["applied_keys"] = list(flat_overrides.keys())

    for key, value in flat_overrides.items():
        # hard-block mask/shift edits from JSON
        if key in DISALLOWED_MASK_KEYS:
            print(f"⚠️  Ignoring override for '{key}' (mask/shift keys are not settable via JSON).")
            continue

        if key in ALLOWED_OVERRIDES:
            try:
                if key in ("MASK_GATING_PASS1", "MASK_GATING_PASS2") and value not in ("soft", "hard"):
                    print(f" Invalid value for {key}: {value}; must be 'soft' or 'hard'")
                    continue
                setattr(config, key, ALLOWED_OVERRIDES[key](value))
            except (ValueError, TypeError) as e:
                print(f" Invalid override for {key}: {value} ({e})")
        else:
            print(f" Unknown override key: {key}")

    if flat_overrides:
        print(f"[OVERRIDES] preset: {preset_name} ({info['enabled_count']} key(s) from {info['source']}: {info['applied_keys']})")
    else:
        print(f" Overrides: empty JSON at {path}, preset: {preset_name}")

    return info


# def list_mkvs(session_root: Path, max_files: int, recursive: bool) -> List[Path]:
    # """List .mkv files in session_root."""
    # pattern = "**/*.mkv" if recursive else "*.mkv"
    # return sorted(session_root.glob(pattern))[:max_files]

def list_mkvs(video_root: Path, max_files: int, recursive: bool) -> List[Path]:
    """List .mkv files in video_root."""
    pattern = "**/*.mkv" if recursive else "*.mkv"
    return sorted(video_root.glob(pattern))[:max_files]

# replaced as nec to set new path for OF 
# def get_paths(video_path: str, module_name: str) -> Dict[str, str]:
    # """Generate paths for logs and outputs."""
    # video_path = Path(video_path)
    # log_dir = video_path.parent / f"{module_name}_logs"
    # return {
        # "LOG_DIR": str(log_dir),
        # "VIDEO_OUT": str(video_path.parent / f"{video_path.stem}_processed.mp4"),
        # "SEGMENT_DIR": str(log_dir / "segments"),
    # }

def get_paths(*args, **kwargs):
    raise RuntimeError("Local get_paths() is deprecated. Use std_paths_of.get_paths(...) instead.")




def get_axis_ptz_snapshot(axis_ip: str = None, axis_creds_path: str = None) -> Dict[str, Any]:
    """Fetch PTZ snapshot from Axis camera."""
    try:
        axis_ip = axis_ip or AXIS_IP_DEFAULT
        axis_creds_path = axis_creds_path or AXIS_CREDS_PATH_DEFAULT
        with open(axis_creds_path, "r", encoding="utf-8-sig") as f:
            creds = json.load(f)
        auth = HTTPDigestAuth(creds["username"], creds["password"])
        r = requests.get(f"http://{axis_ip}/axis-cgi/com/ptz.cgi?query=position", auth=auth, timeout=3)
        r.raise_for_status()
        text = r.text
        snap: Dict[str, str] = {}
        for line in text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                snap[k.strip()] = v.strip()

        def _f(k, cast=float, default=None):
            try:
                return cast(snap.get(k))
            except Exception:
                return default

        return {
            "preset": "Home",  # TODO: Determine preset from PTZ values
            "pan_deg": _f("pan", float, 0.0),
            "tilt_deg": _f("tilt", float, 0.0),
            "zoom": _f("zoom", int, 1),
        }
    except Exception as e:
        print(f" PTZ snapshot error: {e}")
        return {"preset": "Home", "pan_deg": 0.0, "tilt_deg": 0.0, "zoom": 1}


def _fetch_axis_snapshot(axis_ip: str, axis_creds_path: str) -> np.ndarray:
    """Fetch a snapshot from an Axis camera."""
    try:
        axis_ip = axis_ip or AXIS_IP_DEFAULT
        axis_creds_path = axis_creds_path or AXIS_CREDS_PATH_DEFAULT
        with open(axis_creds_path, "r", encoding="utf-8-sig") as f:
            creds = json.load(f)
        auth = HTTPDigestAuth(creds["username"], creds["password"])
        r = requests.get(f"http://{axis_ip}/axis-cgi/jpg/image.cgi", auth=auth, timeout=3)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None else np.zeros((1080, 1920, 3), dtype=np.uint8)
    except Exception as e:
        print(f" Snapshot error: {e}")
        return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _load_reference_still(preset: str, ref_dir: Path) -> np.ndarray:
    """Load a reference still image from \\frame_metadata."""
    image_path = ref_dir / preset / "image"  # Adjust extension if needed
    img = cv2.imread(str(image_path))
    if img is not None:
        return img
    print(f" Failed to load reference image from {image_path}")
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _extract_frame_from_clip(video_path: str, frame_index: str) -> Tuple[np.ndarray, int]:
    """Extract a frame from a video clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((1080, 1920, 3), dtype=np.uint8), 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = total_frames // 2 if frame_index == "middle" else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else np.zeros((1080, 1920, 3), dtype=np.uint8)), idx


def get_preview_image(
    source: str,
    clip_path: Path,
    ref_dir: Path,
    ref_preset: str,
    axis_ip: str,
    axis_creds: str,
) -> Tuple[np.ndarray, str]:
    """Return (image, desc) for the requested preview source."""
    if source == "live":
        img = _fetch_axis_snapshot(axis_ip or AXIS_IP_DEFAULT, axis_creds or AXIS_CREDS_PATH_DEFAULT)
        return img, f"live@{axis_ip or AXIS_IP_DEFAULT}"

    if source == "downloads":
        # newest snapshot_*.jpg in Downloads
        files = sorted(DOWNLOADS_DIR.glob(DOWNLOAD_SNAPSHOT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            path = files[0]
            img = cv2.imread(str(path))
            if img is None:
                print(f"⚠️  Could not read snapshot: {path}")
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            return img, f"downloads:{path.name}"
        else:
            print(f"⚠️  No files matching {DOWNLOAD_SNAPSHOT_GLOB} in {DOWNLOADS_DIR}")
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            return img, "downloads:<none>"

    if source == "reference":
        img = _load_reference_still(ref_preset, ref_dir)
        return img, f"reference:{ref_preset}"

    # default → clip
    img, fi = _extract_frame_from_clip(str(clip_path), "middle")
    return img, f"clip:{clip_path.name}@frame={fi}"

# gs new fix 6
def _draw_dashed_rect(img, p1, p2, color, thickness=2, dash_len=12, gap_len=8):
    x1, y1 = p1; x2, y2 = p2
    x1, x2 = int(min(x1, x2)), int(max(x1, x2))
    y1, y2 = int(min(y1, y2)), int(max(y1, y2))

    def _line(a, b):
        ax, ay = a; bx, by = b
        L = int(np.hypot(bx - ax, by - ay))
        if L <= 0: return
        dx, dy = (bx - ax) / L, (by - ay) / L
        pos = 0
        while pos < L:
            sx = int(ax + dx * pos);  sy = int(ay + dy * pos)
            ex = int(ax + dx * min(pos + dash_len, L))
            ey = int(ay + dy * min(pos + dash_len, L))
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_8)
            pos += dash_len + gap_len

    _line((x1, y1), (x2, y1)); _line((x1, y2), (x2, y2))
    _line((x1, y1), (x1, y2)); _line((x2, y1), (x2, y2))


def _overlay_bboxes(frame, mask, *, min_area=100, thickness=2, dash_len=12, gap_len=8, color=(0,255,0)):
    if mask is None: 
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= int(min_area):
            x, y, w, h = cv2.boundingRect(c)
            _draw_dashed_rect(frame, (x, y), (x + w, y + h),
                              color, thickness=thickness, dash_len=dash_len, gap_len=gap_len)



def draw_preview_caption(img: np.ndarray, config: Config, state: State) -> None:
    """
    Draw captions and visual overlays for enabled filters.
    Uses _water_polygon(preset_data, config, frame_w, frame_h) so water-mask shifts
    affect Preview and runtime identically, regardless of the current image size.
    """
    # --- locals / fonts ---
    margin = int(getattr(config, "CAPTION_MARGIN", 12))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = float(getattr(config, "CAPTION_FONTSCALE", 0.6))
    th     = int(getattr(config, "CAPTION_THICKNESS", 1))

    preset   = state.requested_preset or "custom"
    live_ptz = state.camera_snapshot or {"pan_deg": 0, "tilt_deg": 0, "zoom": 0}

    # --- Row 1: Camera settings (white) ---
    if state.camera_snapshot is None:
        row1 = f"Preset Settings: {preset}"
    else:
        row1 = (
            f"Camera Settings: {preset} | "
            f"pan = {live_ptz['pan_deg']} | tilt = {live_ptz['tilt_deg']} | zoom = {live_ptz['zoom']}"
        )
    y1 = margin + 16
    cv2.putText(img, row1, (margin, y1), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, row1, (margin, y1), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    # --- Row 2: Mode (white) ---
    ov_mode = (state.override_info or {}).get("mode", "None")
    row2 = f"Mode: {ov_mode}"
    y2 = margin + 32
    cv2.putText(img, row2, (margin, y2), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, row2, (margin, y2), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    # --- Green sections start here (row 3+) ---
    y = margin + 48

    # --- Resolve preset data safely ---
    preset_data   = CAMERA_PRESETS.get(preset) or {}
    water_profile = preset_data.get("water_profile") or []

    # ROIs come from the active camera preset (keeps Preview == Runtime)
    rois       = list((preset_data.get("rois") or []))
    rois_names = list((preset_data.get("rois_names") or []))

    # Keep names aligned with rois length (pad or trim)
    if len(rois_names) != len(rois):
        if len(rois_names) < len(rois):
            rois_names += [f"roi_{i}" for i in range(len(rois_names), len(rois))]
        else:
            rois_names = rois_names[:len(rois)]

            
            

    # --- WATER legend + translucent overlay (green) ---
        # --- WATER legend + translucent overlay (green) ---
    if bool(getattr(config, "WATER_MASK_ENABLED", False)) and water_profile:
        legend = f"WATER: {getattr(config, 'WATER_MASK_MODE', 'exclude')} | pts={len(water_profile)}"
        cv2.putText(img, legend, (margin, y), font, fs, (40, 255, 40), 2, cv2.LINE_AA)
        y += 20

        # Build shifted polygon in native 1920x1080 space
        poly, npts = _water_polygon(preset_data, config)
        if poly is not None and int(npts) >= 3:
            # Scale to current frame size (handles Downloads JPG, reference stills, etc.)
            h_img, w_img = img.shape[:2]
            sx = w_img / 1920.0
            sy = h_img / 1080.0
            poly_scaled = poly.copy()
            poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
            poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_scaled], color=(0, 255, 0))
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0.0, dst=img)

        
    if bool(getattr(config, "ROI_FILTER_ENABLED", False)) and rois:
        legend = f"ROI: {getattr(config, 'ROI_FILTER_MODE', 'exclude')} ({len(rois)} ROIs)"
        cv2.putText(img, legend, (margin, y), font, fs, (40, 255, 40), 2, cv2.LINE_AA)
        y += 20

        # Scale ROIs from the design resolution (1920x1080) to the current image size
        h, w = img.shape[:2]
        BASE_W, BASE_H = 1920, 1080
        sx = w / float(BASE_W)
        sy = h / float(BASE_H)

        for (x1, y1b, x2, y2b), name in zip(rois, rois_names):
            # scale
            X1 = int(round(x1 * sx)); Y1 = int(round(y1b * sy))
            X2 = int(round(x2 * sx)); Y2 = int(round(y2b * sy))
            # clip to bounds
            X1 = max(0, min(w - 1, X1)); X2 = max(0, min(w - 1, X2))
            Y1 = max(0, min(h - 1, Y1)); Y2 = max(0, min(h - 1, Y2))
            # normalize ordering
            if X2 < X1: X1, X2 = X2, X1
            if Y2 < Y1: Y1, Y2 = Y2, Y1
            # draw
            cv2.rectangle(img, (X1, Y1), (X2, Y2), (255, 0, 0), 2)
            cv2.putText(img, str(name), (X1, max(0, Y1 - 10)), font, fs, (255, 0, 0), 1, cv2.LINE_AA)

    

# gs
def build_preview_with_options(
    video_root: Path,
    first_clip_path: Path,
    log_dir: Path,
    default_source: str,
    reference_dir: Path,
    reference_preset: str,
    axis_ip: Optional[str],
    axis_creds_path: Optional[str],
    clip_frame_index: str,
    auto_open: bool,
    config: Config,
    state: State,
) -> Path:
    """
    Generate and save a preview image with captions.

    Fixes:
      - When source is Reference or Clip, we now explicitly select (or keep) a camera preset
        and write it to state.requested_preset so mask profiles & shifts apply correctly.
      - Live keeps using the actual camera snapshot; requested_preset is derived from snapshot
        if available (or left as-is).
    """
    # --- choose source ---------------------------------------------------------
        # --- choose source ---------------------------------------------------------
    while True:
        try:
            raw = input("Select preview source: [1] Live, [2] Axis snapshot (Downloads), [3] Clip, [4] Reference: ").strip()
            if raw in ("1", "2", "3", "4"):
                source_choice = int(raw)
                break
            print("⚠️  Invalid input. Please enter 1, 2, 3, or 4.")
        except (KeyboardInterrupt, EOFError):
            print("🛑 Aborted by user.")
            sys.exit(0)

    source_map = {1: "live", 2: "downloads", 3: "clip", 4: "reference"}
    source = source_map[source_choice]


    # --- resolve preset per source --------------------------------------------
    if source == "live":
        # Pull live snapshot; if it contains a preset name, prefer that.
        snap = get_axis_ptz_snapshot(axis_ip=axis_ip, axis_creds_path=axis_creds_path)
        state.camera_snapshot = snap
        snap_preset = (snap.get("preset") or "").strip()
        if snap_preset:
            state.requested_preset = snap_preset
        elif not state.requested_preset:
            state.requested_preset = "Home"  # safe default
    else:
        # Reference / Clip: we are NOT using live PTZ; clear snapshot
        state.camera_snapshot = None

        # Ask which preset to render masks against; default to current or Home
        preset_keys = list(CAMERA_PRESETS.keys())
        default_preset = state.requested_preset if state.requested_preset in preset_keys else "Home"
        print(f"Available camera presets: {', '.join(preset_keys)}")
        sel = input(f"Choose preset for mask/caption [{default_preset}]: ").strip()
        chosen = sel if sel in preset_keys else default_preset
        state.requested_preset = chosen

    # --- get preview frame -----------------------------------------------------
    img, desc = get_preview_image(
        source=source,
        clip_path=first_clip_path,
        ref_dir=reference_dir,
        ref_preset=reference_preset,
        axis_ip=axis_ip or AXIS_IP_DEFAULT,
        axis_creds=axis_creds_path or AXIS_CREDS_PATH_DEFAULT,
    )
    
    # Tiny sanity echo (optional while you tune).
    print(f"[SHIFT] X={config.WATER_SHIFT_X}  YT={config.WATER_SHIFT_Y_TOP}  YB={config.WATER_SHIFT_Y_BOTTOM}  dilate={config.WATER_MASK_DILATE_PX}")

    # --- draw overlays/captions (uses state.requested_preset + config shifts) --
    draw_preview_caption(img, config, state)

    # --- save & optionally open ------------------------------------------------
    safe_desc = desc.replace(":", "_")
    out_path = (log_dir / f"preview_{safe_desc}.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"👀 Preview saved → {out_path}")
    print(f"[PREVIEW] source={source} preset={state.requested_preset}")

    if auto_open:
        try:
            import os
            os.startfile(str(out_path))  # Windows
        except Exception:
            pass

    return out_path



#  To be used w/ Axis camera feature: take photo / download
def _load_latest_download_snapshot(downloads_dir: Path = DOWNLOADS_DIR_DEFAULT):
    """
    Return (img, desc) for the newest 'snapshot_*.jpg' in Downloads.
    If none found or unreadable, returns (None, reason_str).
    """
    pattern = str(downloads_dir / "snapshot_*.jpg")
    files = glob.glob(pattern)
    if not files:
        return None, f"No snapshot_*.jpg in {downloads_dir}"
    latest = max(files, key=os.path.getmtime)
    img = cv2.imread(latest)
    if img is None:
        return None, f"Failed to read: {latest}"
    return img, f"downloads:{Path(latest).name}"



def _print_active_filter_settings(config: Config, state: State, camera_preset: str) -> None:

    """Print active filter settings."""
    preset = state.requested_preset or "custom"
    preset_data = CAMERA_PRESETS.get(preset, {"water_profile": [], "rois": [], "rois_names": []})
    
    print("\nActive Settings Summary:")
    print(f"  Camera Preset: {camera_preset}")
    print("    → affects mask geometry, water profile, caption overlays")
    print("  Knob Source: Config defaults")
    print("    → no CLI or JSON overrides detected")
    print("  Key Gating Parameters:")
    print(f"    MOTION_THRESHOLD: {config.MOTION_THRESHOLD}")
    print(f"    MIN_MOTION_PIXELS: {config.MIN_MOTION_PIXELS}")
    print(f"    MASK_GATING_PASS1: {config.MASK_GATING_PASS1}")
    print(f"    MASK_GATING_PASS2: {config.MASK_GATING_PASS2}")

    
    
    if config.ROI_FILTER_ENABLED:
        # print(f"  ROI_FILTER: {config.ROI_FILTER_MODE} ({len(preset_data['rois'])} ROIs)")
        active_rois = globals().get("ROIS", [])
    if config.WATER_MASK_ENABLED:
        print(f"  WATER_MASK: {config.WATER_MASK_MODE} ({len(preset_data['water_profile'])} points)")


def build_keep_mask(config: Config, frame_shape: Tuple[int, int], preset: str) -> np.ndarray:
    """Build a mask combining ROI (wind) and water filters."""
    
    poly, npts = None, 0  # ← ensure defined even when water mask is disabled/empty

    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    preset_data = CAMERA_PRESETS.get(preset, {"water_profile": [], "rois": []})
    rois = preset_data.get("rois") or []

    # --- ROI (wind) mask ------------------------------------------------------
    if config.ROI_FILTER_ENABLED and rois:
        mode = (config.ROI_FILTER_MODE or "exclude").lower()
        if mode == "include":
            # Keep only the union of ROIs
            mask[:, :] = 0
            for (x1, y1, x2, y2) in rois:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                mask[y1:y2, x1:x2] = 255
        else:
            # Exclude each ROI region
            for (x1, y1, x2, y2) in rois:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                mask[y1:y2, x1:x2] = 0

    # --- Water mask -----------------------------------------------------------
    """
    if config.WATER_MASK_ENABLED and preset_data["water_profile"]:
        poly, npts = _water_polygon(preset_data, config)
    if poly is not None and int(npts) >= 3:
        # Scale to this frame's size
        frame_h, frame_w = frame_shape[:2]
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        poly_scaled = poly.copy()
        poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
        poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

        water_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(water_mask, [poly_scaled], 255)

        if config.WATER_MASK_DILATE_PX > 0:
            k = int(config.WATER_MASK_DILATE_PX)
            kernel = np.ones((k, k), np.uint8)
            water_mask = cv2.dilate(water_mask, kernel)

        if config.WATER_MASK_MODE == "exclude":
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(water_mask))
        else:
            mask = cv2.bitwise_and(mask, water_mask)
        """   
            
    # --- Water mask -----------------------------------------------------------
    # Always initialize to avoid UnboundLocalError when water mask is off or preset has no profile
    poly, npts = (None, 0)
    if config.WATER_MASK_ENABLED and preset_data.get("water_profile"):
        poly, npts = _water_polygon(preset_data, config)

    if poly is not None and int(npts) >= 3:
        # Scale to this frame's size
        frame_h, frame_w = frame_shape[:2]
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        poly_scaled = poly.copy()
        poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
        poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

        water_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(water_mask, [poly_scaled], 255)

        if int(getattr(config, "WATER_MASK_DILATE_PX", 0)) > 0:
            k = int(config.WATER_MASK_DILATE_PX)
            kernel = np.ones((k, k), np.uint8)
            water_mask = cv2.dilate(water_mask, kernel)

        if (getattr(config, "WATER_MASK_MODE", "exclude") or "exclude") == "exclude":
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(water_mask))
        else:
            mask = cv2.bitwise_and(mask, water_mask)
            
    return mask


# --- New - Per-frame debug logger (CSV) --------------------------------------------

# ==== PerFrameLogger (drop-in replacement / edits) ====
# ===== PerFrameLogger (refactored) ==========================================
class PerFrameLogger:
    def __init__(self, path: Path, total_frames: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        self.w.writerow(["pass","frame_idx","active_raw","active_gated","kept","flow_dx_mean","flow_speed_mean"])
        self._closed = False
        self.total_frames: Optional[int] = int(total_frames) if total_frames is not None else None
        self._max_frame_idx_seen = 0
        self._p1_kept = 0
        self._p2_kept = 0

    def log_p1(self, frame_idx: int, active_raw: int, active_gated: int, kept: bool,
               flow_dx_mean: float, flow_speed_mean: float):
        self._max_frame_idx_seen = max(self._max_frame_idx_seen, int(frame_idx))
        if kept:
            self._p1_kept += 1
        self.w.writerow(["p1", int(frame_idx), int(active_raw), int(active_gated),
                         bool(kept), float(flow_dx_mean), float(flow_speed_mean)])

    def log_p2(self, frame_idx: int, active_raw: int, active_gated: int, kept: bool,
               flow_dx_mean: float, flow_speed_mean: float):
        self._max_frame_idx_seen = max(self._max_frame_idx_seen, int(frame_idx))
        if kept:
            self._p2_kept += 1
        self.w.writerow(["p2", int(frame_idx), int(active_raw), int(active_gated),
                         bool(kept), float(flow_dx_mean), float(flow_speed_mean)])

    def _write_summary(self):
        # Optional: compute totals safely for any summary you want here.
        tf = self.total_frames if self.total_frames is not None else self._max_frame_idx_seen
        _ = int(tf) if tf else 0
        # (Write summary rows if desired; otherwise leave empty.)

    def close(self):
        if self._closed:
            return
        try:
            self._write_summary()
        finally:
            self.fh.close()
            self._closed = True
# ============================================================================ 

# ==== end PerFrameLogger edits ====

    
# ---      End --------------------------------------------


def _segments_from_kept_frames(kept_frames: List[int], config: Config) -> List[Tuple[int, int]]:
    """Merge kept frame indices into segments using MERGE_GAP_FRAMES / MIN_SEGMENT_LENGTH."""
    if not kept_frames:
        return []
    kept_frames = sorted(int(x) for x in kept_frames)
    segments: List[Tuple[int, int]] = []
    start = prev = kept_frames[0]
    for f in kept_frames[1:]:
        if f <= prev + int(config.MERGE_GAP_FRAMES):
            prev = f
            continue
        if (prev - start + 1) >= int(config.MIN_SEGMENT_LENGTH):
            segments.append((start, prev))
        start = prev = f
    if (prev - start + 1) >= int(config.MIN_SEGMENT_LENGTH):
        segments.append((start, prev))
    return segments


def _segment_span_len(seg: Tuple[int, int]) -> int:
    return int(seg[1] - seg[0] + 1)


def _count_kept_in_span(kept: set, start: int, end: int) -> int:
    """Count kept frame indices within [start,end] inclusive. kept is a set of int frame indices."""
    if not kept:
        return 0
    # kept is a set; counting via iteration is fine at typical clip lengths.
    s = int(start)
    e = int(end)
    return sum(1 for k in kept if s <= int(k) <= e)


def _segment_density(kept: set, seg: Tuple[int, int]) -> Tuple[int, int, float]:
    """Return (md_count, span_len, density) for a segment given a kept-set."""
    span_len = _segment_span_len(seg)
    md_count = _count_kept_in_span(kept, seg[0], seg[1])
    density = (md_count / span_len) if span_len > 0 else 0.0
    return int(md_count), int(span_len), float(density)


def _longest_by_span(segments: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Return the segment with the largest span length (ties: earlier start)."""
    if not segments:
        return None
    return max(segments, key=lambda se: (_segment_span_len(se), -int(se[0])))


def detect_motion_frames(
    video_path: Path,
    config: Config,
    preset: str,
    logger: 'PerFrameLogger' = None
) -> Tuple[List[Tuple[int, int]], set, int]:
    """Pass-1: scan full clip, compute motion using P1 thresholds, and return (segments, kept_set, total_frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" Failed to open video: {video_path}")
        return [], set(), 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        return [], set(), total_frames

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    kept_p1: set = set()

    for idx in range(1, total_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        raw_mask = (mag > float(config.MOTION_THRESHOLD)).astype(np.uint8) * 255
        active_raw = int(cv2.countNonZero(raw_mask))

        if config.MASK_GATING_PASS1 == "hard":
            motion_mask = cv2.bitwise_and(raw_mask, keep_mask)
        else:
            motion_mask = raw_mask

        active_gated = int(cv2.countNonZero(motion_mask))
        kept = (active_gated > int(config.MIN_MOTION_PIXELS))

        flow_dx_mean = 0.0
        flow_speed_mean = 0.0
        if active_gated > 0:
            m = motion_mask > 0
            flow_dx_mean = float(np.mean(flow[..., 0][m]))
            flow_speed_mean = float(np.mean(mag[m]))

        if logger is not None:
            # Existing logger expects (frame_idx, raw, gated, kept, dx_mean, speed_mean)
            logger.log_p1(idx, active_raw, active_gated, kept, flow_dx_mean, flow_speed_mean)

        if kept:
            kept_p1.add(idx)

        prev_gray = gray

        if idx % STATUS_EVERY == 0:
            print(f"   ▶ Frame {idx:05d}/{total_frames} | raw={active_raw} gated={active_gated} kept={1 if kept else 0}")

    cap.release()

    segments = _segments_from_kept_frames(sorted(kept_p1), config)
    return segments, kept_p1, total_frames


def p2_recover_and_rebuild_segments(
    p1_segments: List[Tuple[int, int]],
    kept_p1: set,
    video_path: Path,
    config: Config,
    preset: str,
    log_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[int, int]], set, set, Dict[str, Any]]:
    """
    Pass-2 recovery: for each P1 segment, scan a padded window using P2 thresholds and Pass-2 gating.
    Produce recovered MD frames (kept_p2), combine with kept_p1 to form kept_final, and rebuild segments from kept_final.
    Optionally writes a recovery audit CSV to log_dir.
    """
    if not p1_segments:
        return [], set(), set(kept_p1), {
            "p1_segments": 0, "p2_windows": 0, "p2_recovered_frames": 0, "p2_recovered_in_pad": 0,
            "final_segments": 0, "final_frames": int(len(kept_p1)),
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" Failed to open video for P2 recovery: {video_path}")
        return p1_segments, set(), set(kept_p1), {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

    pad = int(getattr(config, "P2_PAD_FRAMES", 0) or 0)
    thr2 = float(getattr(config, "MOTION_THRESHOLD_P2", config.MOTION_THRESHOLD))
    pix2 = int(getattr(config, "MIN_MOTION_PIXELS_P2", config.MIN_MOTION_PIXELS))

    kept_p2: set = set()
    recovered_in_pad = 0

    # Prepare audit CSV (only frames we evaluate in P2 windows)
    audit_rows = []
    def _clamp(a: int) -> int:
        return max(0, min(int(a), total_frames - 1))

    for (s, e) in p1_segments:
        win_s = _clamp(s - pad)
        win_e = _clamp(e + pad)
        if win_e <= win_s:
            continue

        # Seek to win_s and prime prev_gray as frame at win_s
        cap.set(cv2.CAP_PROP_POS_FRAMES, win_s)
        ok, prev = cap.read()
        if not ok or prev is None:
            continue
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for idx in range(win_s + 1, win_e + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            raw_mask = (mag > thr2).astype(np.uint8) * 255
            active_raw = int(cv2.countNonZero(raw_mask))

            if config.MASK_GATING_PASS2 == "hard":
                motion_mask = cv2.bitwise_and(raw_mask, keep_mask)
            else:
                motion_mask = raw_mask

            active_gated = int(cv2.countNonZero(motion_mask))

            # P2-final per-frame energy (sum of flow magnitude inside the gated P2 mask)
            # This is the primary cadence-bearing signal; post-analysis can smooth/FFT later.
            if active_gated > 0:
                flow_energy_p2 = float(mag[motion_mask > 0].sum())
                flow_energy_norm_p2 = float(flow_energy_p2 / max(active_gated, 1))
            else:
                flow_energy_p2 = 0.0
                flow_energy_norm_p2 = 0.0

            kept = (active_gated > pix2)

            in_p1_span = int(s <= idx <= e)
            in_pad = int((idx < s) or (idx > e))

            if kept and (idx not in kept_p1):
                kept_p2.add(idx)
                if in_pad:
                    recovered_in_pad += 1

            audit_rows.append((idx, in_p1_span, in_pad,
                               int(idx in kept_p1), int(kept), int((idx in kept_p1) or kept),
                               active_raw, flow_energy_p2, flow_energy_norm_p2))

            prev_gray = gray

    cap.release()

    kept_final = set(kept_p1) | set(kept_p2)
    final_segments = _segments_from_kept_frames(sorted(kept_final), config)

    stats = {
        "p1_segments": int(len(p1_segments)),
        "p2_windows": int(len(p1_segments)),
        "p1_frames": int(len(kept_p1)),
        "p2_recovered_frames": int(len(kept_p2)),
        "p2_recovered_in_pad": int(recovered_in_pad),
        "final_frames": int(len(kept_final)),
        "final_segments": int(len(final_segments)),
        "gain_frames": int(len(kept_final) - len(kept_p1)),
    }

    if log_dir is not None:
        try:
            out = Path(log_dir) / "flow_p2_frame_log.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx","in_p1_span","in_p2_pad","md_p1","md_p2","md_final","active_raw_p2","flow_energy_p2","flow_energy_norm_p2"])
                w.writerows(audit_rows)
                
                # gs
                # --- standardized packet export (OF) ---
                try:
                    run_dir = Path(log_dir).parent  # logs/ -> <run_dir>/
                    export_comparison_packet_of(
                        run_dir=run_dir,
                        video_path=video_path,
                        fps=fps,
                        total_frames=total_frames,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        config=config,
                        state=state,
                        audit_rows=audit_rows,
                        audit_header=["frame_idx","in_p1_span","in_p2_pad","md_p1","md_p2","md_final","active_raw_p2","flow_energy_p2","flow_energy_norm_p2"],
                        final_segments=final_segments,
                        p1_segments=p1_segments,
                    )
                except Exception as e:
                    print(f"   ⚠️ packet export failed: {e}")
                
        except Exception as e:
            print(f"   ⚠️ failed to write P2 recovery audit CSV: {e}")

    print(f"   ✓ P2 recovery: +{stats['gain_frames']} frame(s) (P2-only={stats['p2_recovered_frames']}, in_pad={stats['p2_recovered_in_pad']}); final_segments={stats['final_segments']}")
    return final_segments, kept_p2, kept_final, stats
def draw_dashed_rect(img, p1, p2, color, thickness=2, dash_len=12, gap_len=8):
    """Draw a dashed rectangle from p1(x1,y1) to p2(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    x1, x2 = int(min(x1, x2)), int(max(x1, x2))
    y1, y2 = int(min(y1, y2)), int(max(y1, y2))

    def _dashed_line(pt1, pt2):
        xA, yA = pt1
        xB, yB = pt2
        length = int(np.hypot(xB - xA, yB - yA))
        if length == 0:
            return
        dx = (xB - xA) / length
        dy = (yB - yA) / length
        pos = 0
        while pos < length:
            x_start = int(xA + dx * pos)
            y_start = int(yA + dy * pos)
            x_end   = int(xA + dx * min(pos + dash_len, length))
            y_end   = int(yA + dy * min(pos + dash_len, length))
            cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness, lineType=cv2.LINE_8)
            pos += dash_len + gap_len

    _dashed_line((x1, y1), (x2, y1))  # top
    _dashed_line((x1, y2), (x2, y2))  # bottom
    _dashed_line((x1, y1), (x1, y2))  # left
    _dashed_line((x2, y1), (x2, y2))  # right


def overlay_bboxes(frame, mask, min_area: int, box_thickness: int, dash_len: int, gap_len: int):
    """
    Find contours on 'mask' and draw dashed boxes for blobs >= min_area.
    Uses global style knobs already present in Config: BOX_OVERLAY controls call site.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= int(min_area):
            x, y, w, h = cv2.boundingRect(cnt)
            draw_dashed_rect(
                frame,
                (x, y),
                (x + w, y + h),
                color=(0, 255, 0),
                thickness=int(box_thickness),
                dash_len=int(dash_len),
                gap_len=int(gap_len),
            )

#  =========================== End - Restore the bound boxes =======================================================

def save_run_outputs(
    log_dir: Path,
    config: Config,
    state: State,
    video_path: Path,
    segments: List[Tuple[int, int]],
    duration: float,
    fps: float,
    total_frames: int,
    frame_w: int,
    frame_h: int,
) -> None:
    """Save logs and configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    seg_log = log_dir / "flow_segment_log.csv"
    master = log_dir / "flow_segment_log_master.csv"
    overrides_log = log_dir / "overrides_log.txt"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(i, s, e, (e - s + 1) / fps) for i, (s, e) in enumerate(segments, 1)]

    with open(seg_log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "end_frame", "duration_sec"])
        w.writerows(rows)

    with open(master, "a", newline="", encoding="utf-8") as f:
        f.write(f"# Run at {now}\n")
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "end_frame", "duration_sec"])
        w.writerows(rows)
        f.write(f"# Processing time: {duration:.2f} seconds\n\n")

    with open(overrides_log, "a", encoding="utf-8") as f:
        f.write(f"Run at {now}\n")
        f.write(f"Preset: {state.override_info['preset_name']}\n")
        f.write(f"Overrides: {json.dumps(state.override_info['values'], indent=2)}\n\n")

    # collect info and write json file (run_config.json)
    config_json = log_dir / "run_config.json"
    payload = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "timestamp": now,
        "clip_info": {"fps": float(fps), "total_frames": int(total_frames), "w": int(frame_w), "h": int(frame_h)},
        "knobs": {k: v for k, v in config.__dict__.items() if k in ALLOWED_OVERRIDES},
        "overrides": state.override_info,
        "camera_preset": state.requested_preset or "custom",
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)




# ============================== new output folder to compare algorithms =====================


def export_comparison_packet_of(
    *,
    run_dir: Path,
    video_path: Path,
    fps: float,
    total_frames: int,
    frame_w: int,
    frame_h: int,
    config: "Config",
    state: "State",
    audit_rows: List[List[Any]],
    audit_header: List[str],
    final_segments: List[Tuple[int, int]],
    p1_segments: List[Tuple[int, int]] | None = None,
) -> None:
    """
    Write standardized comparison outputs:
      packet/run_manifest.json
      packet/frame_signal.csv
      packet/segment_summary.csv

    Uses OF P2 audit_rows already computed during P2 recovery.
    """
    packet_dir = Path(run_dir) / "packet"
    packet_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) run_manifest.json
    # -------------------------
    manifest = {
        "pipeline_name": "OF_HO_P2",
        "pipeline_version": str(SCRIPT_VERSION),
        "module": str(MODULE_NAME),
        "video_name": video_path.name,
        "video_path": str(video_path),
        "clip_info": {"fps": float(fps), "total_frames": int(total_frames), "w": int(frame_w), "h": int(frame_h)},
        "override_info": getattr(state, "override_info", None),
        "requested_preset": getattr(state, "requested_preset", None),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (packet_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -------------------------
    # 2) Determine primary segment (FINAL longest span)
    # -------------------------
    primary = None
    if final_segments:
        # "longest span" winner
        primary = max(final_segments, key=lambda se: (se[1] - se[0] + 1))
    primary_s, primary_e = primary if primary else (None, None)

    # -------------------------
    # 3) frame_signal.csv (standard schema)
    # -------------------------
    # We expect audit_header to include at least:
    # frame_idx, md_final, active_raw_p2, flow_energy_p2, flow_energy_norm_p2
    # If your header differs, map by name (not position).
    col_idx = {name: i for i, name in enumerate(audit_header)}

    required = ["frame_idx", "md_final", "active_raw_p2", "flow_energy_p2", "flow_energy_norm_p2"]
    missing = [c for c in required if c not in col_idx]
    if missing:
        print(f"   ⚠️ packet export skipped: missing audit columns: {missing}")
        return

    out_rows = []
    for r in audit_rows:
        fidx = int(r[col_idx["frame_idx"]])
        md_flag = int(r[col_idx["md_final"]])
        active = r[col_idx["active_raw_p2"]]
        e_raw = r[col_idx["flow_energy_p2"]]
        e_norm = r[col_idx["flow_energy_norm_p2"]]

        in_primary = 0
        if primary_s is not None:
            in_primary = 1 if (primary_s <= fidx <= primary_e) else 0

        out_rows.append([
            fidx,                 # frame_idx
            md_flag,              # md_flag
            in_primary,           # in_primary_span
            active,               # active_pixels
            e_raw,                # energy_raw
            e_norm,               # energy_norm
            "",                   # x
            "",                   # y
            "OF",                 # source
        ])

    frame_signal_path = packet_dir / "frame_signal.csv"
    with frame_signal_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "md_flag",
            "in_primary_span",
            "active_pixels",
            "energy_raw",
            "energy_norm",
            "x",
            "y",
            "source",
        ])
        w.writerows(out_rows)

    # -------------------------
    # 4) segment_summary.csv (standard schema)
    # -------------------------
    # Compute md_total/density/gaps using the per-frame md_flag within each segment span.
    md_by_frame = {row[0]: int(row[1]) for row in out_rows}  # frame_idx -> md_flag

    def _gap_stats(md_flags: List[int]) -> tuple[int, int]:
        """Return (gap_count, max_gap_len) for 0-runs inside a span."""
        gap_count = 0
        max_gap = 0
        cur = 0
        for v in md_flags:
            if v == 0:
                cur += 1
            else:
                if cur > 0:
                    gap_count += 1
                    max_gap = max(max_gap, cur)
                    cur = 0
        if cur > 0:
            gap_count += 1
            max_gap = max(max_gap, cur)
        return gap_count, max_gap

    seg_rows = []
    for i, (s, e) in enumerate(final_segments or [], 1):
        span = int(e - s + 1)
        md_flags = [md_by_frame.get(fi, 0) for fi in range(s, e + 1)]
        md_total = int(sum(md_flags))
        density = float(md_total / span) if span > 0 else 0.0
        gap_count, max_gap = _gap_stats(md_flags)
        seg_rows.append([
            i,
            s,
            e,
            span,
            md_total,
            density,
            gap_count,
            max_gap,
            1 if primary and (s, e) == primary else 0,
        ])

    seg_sum_path = packet_dir / "segment_summary.csv"
    with seg_sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "segment_id",
            "span_start",
            "span_end",
            "span_len",
            "md_total",
            "density",
            "gap_count",
            "max_gap",
            "is_primary",
        ])
        w.writerows(seg_rows)

    print(f"   📦 packet export → {packet_dir}")






# ================= process clip  =============================================

def process_clip(video_path: Path, paths: Dict[str, str], config: Config, state: State) -> None:
    """Process a single video clip."""
    start_time = datetime.now()

    # --- resolve OF std_paths_of directories & ensure they exist ---
    # std_paths_of uses IMAGE_DIR/VIDEO_DIR; for OF-vs-Step7 comparisons we map to
    # frames/videos to match the Step7 run folder convention.
    _img_dir = Path(paths["IMAGE_DIR"])
    _vid_dir = Path(paths["VIDEO_DIR"])
    log_dir = Path(paths["LOG_DIR"])

    frames_dir = _img_dir.parent / "frames"
    videos_dir = _vid_dir.parent / "videos"

    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    img_dir = frames_dir
    vid_dir = videos_dir

    # --- open the clip first (we need total_frames for the logger) ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" Failed to open video for output: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # --- Per-frame CSV logger (one instance, knows total_frames) ---
    try:
        pfl = PerFrameLogger(log_dir / "flow_per_frame_log.csv", total_frames=total_frames)
    except Exception as e:
        pfl = None
        print(f"   ⚠️ per-frame logger init failed: {e}")

    # --- Pass-1 detect, then Pass-2 recovery (adds MD frames) and rebuild final segments ---
    preset = state.requested_preset or "custom"
    p1_segments, kept_p1, _scan_total = detect_motion_frames(video_path, config, preset, logger=pfl)
    segments, kept_p2, kept_final, p2_stats = p2_recover_and_rebuild_segments(
        p1_segments, kept_p1, video_path, config, preset, log_dir=log_dir
    )

    # --- Terminal truth summary: segment boundaries + true MD counts + density ---
    p1_longest = _longest_by_span(p1_segments)
    final_longest = _longest_by_span(segments)

    p1_total_md = int(len(kept_p1))
    final_total_md = int(len(kept_final))

    if p1_longest is not None:
        p1_md, p1_span, p1_den = _segment_density(kept_p1, p1_longest)
        print(f"   🧩 P1:    n={len(p1_segments)} | longest=({p1_longest[0]}–{p1_longest[1]}) span={p1_span} | md={p1_md} | dens={p1_den:.1%} | md_total={p1_total_md}")
    else:
        print(f"   🧩 P1:    n=0 | md_total={p1_total_md}")

    if final_longest is not None:
        f_md, f_span, f_den = _segment_density(kept_final, final_longest)
        print(f"   🧩 FINAL: n={len(segments)} | longest=({final_longest[0]}–{final_longest[1]}) span={f_span} | md={f_md} | dens={f_den:.1%} | md_total={final_total_md}")
    else:
        print(f"   🧩 FINAL: n=0 | md_total={final_total_md}")

    # Low-density warnings (flag segments whose span is mostly bridged gaps)
    warn_thr = float(getattr(config, "LOW_DENSITY_WARN", 0.20))

    # Optional: print per-segment details (keeps terminal actionable without drowning you)
    def _print_segment_list(tag: str, segs: List[Tuple[int, int]], kept: set):
        if not segs:
            return
        if len(segs) <= 5:
            for (s, e) in segs:
                md_c, span_c, den = _segment_density(kept, (s, e))
                print(f"      • {tag} seg=({s}–{e}) span={span_c} md={md_c} dens={den:.1%}")
            return
        # If many segments, show top-3 by span plus worst density.
        by_span = sorted(segs, key=lambda se: (_segment_span_len(se), -int(se[0])), reverse=True)[:3]
        print(f"      • {tag} segments: showing top-3 by span + worst density (n={len(segs)})")
        for se in by_span:
            md_c, span_c, den = _segment_density(kept, se)
            print(f"        - top span=({se[0]}–{se[1]}) span={span_c} md={md_c} dens={den:.1%}")
        worst = min(segs, key=lambda se: _segment_density(kept, se)[2])
        md_c, span_c, den = _segment_density(kept, worst)
        print(f"        - worst dens=({worst[0]}–{worst[1]}) span={span_c} md={md_c} dens={den:.1%}")

    _print_segment_list("P1", p1_segments, kept_p1)
    _print_segment_list("FINAL", segments, kept_final)
    for tag, segs, kept in (
        ("P1", p1_segments, kept_p1),
        ("FINAL", segments, kept_final),
    ):
        lows = []
        for se in segs:
            md_c, span_c, den = _segment_density(kept, se)
            if span_c > 0 and den < warn_thr:
                lows.append((den, md_c, span_c, se))
        if lows:
            lows.sort(key=lambda x: x[0])
            worst = lows[0]
            den, md_c, span_c, se = worst
            print(
                f"   ⚠️ LOW DENSITY {tag}: worst=({se[0]}–{se[1]}) span={span_c} md={md_c} dens={den:.1%} "
                f"(threshold<{warn_thr:.0%}); consider lowering MERGE_GAP_FRAMES or raising thresholds."
            )

    # DEBUG: lock in the count right after refine (if this shows >0 but final print shows 0,
    # the list is being clobbered later — which we avoid below)
    # print(f"   ✓ P2 returned {len(segments)} segment(s): {segments}")

    # --- compiled video (single reel across all segments) ---
    compiled_writer = None
    if config.SAVE_COMPILED_VIDEO:
        compiled_path = vid_dir / "compiled_flow_video.mp4"
        compiled_writer = cv2.VideoWriter(
            str(compiled_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h)
        )

    # --- emit per-segment videos; also per-frame JPGs + compiled reel ---
    if segments:
        segment_dir = vid_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)

        # build keep mask once for this clip size (Pass-2 gating)
        keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

        # reuse open cap; seek per segment
        for i, (start, end) in enumerate(segments, 1):
            seg_writer = None
            if config.SAVE_PER_SEGMENT_VIDEO:
                seg_path = segment_dir / f"segment_{i}_{video_path.stem}.mp4"
                seg_writer = cv2.VideoWriter(
                    str(seg_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame_w, frame_h)
                )

            # seek and prime
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, frame = cap.read()
            if not ok or frame is None:
                if seg_writer is not None:
                    seg_writer.release()
                continue
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # first frame (no flow yet)
            if config.SAVE_FRAME_JPEGS:
                cv2.imwrite(str(img_dir / f"flow_frame_{start:05d}.jpg"), frame)
            if compiled_writer is not None:
                compiled_writer.write(frame)
            if seg_writer is not None:
                seg_writer.write(frame)

            # frames [start+1 .. end]
            for fidx in range(start + 1, end + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mask = (mag > config.MOTION_THRESHOLD).astype(np.uint8) * 255

                if config.MASK_GATING_PASS2 == "hard":
                    motion_mask = cv2.bitwise_and(motion_mask, keep_mask)

                # overlay (simple rectangles or your dashed overlay helper)
                if config.BOX_OVERLAY:
                    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) >= int(config.BOX_MIN_PIXELS):
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), int(config.BOX_THICKNESS))

                if config.SAVE_FRAME_JPEGS:
                    cv2.imwrite(str(img_dir / f"flow_frame_{fidx:05d}.jpg"), frame)
                if compiled_writer is not None:
                    compiled_writer.write(frame)
                if seg_writer is not None:
                    seg_writer.write(frame)

                prev_gray = gray

            if seg_writer is not None:
                seg_writer.release()

    # tear down writers/capture
    cap.release()
    if compiled_writer is not None:
        compiled_writer.release()

    duration = (datetime.now() - start_time).total_seconds()
    save_run_outputs(Path(paths["LOG_DIR"]), config, state, video_path, segments, duration, fps, total_frames, frame_w, frame_h)

    # close per-frame logger and write summary/metrics inside its close()
    if pfl is not None:
        try:
            pfl.close()
        except Exception as e:
            print(f"   ⚠️ per-frame logger close error: {e}")

    print(f"    Processed {video_path.name}: {len(segments)} segment(s) in {duration:.2f}s")


    # print(f"    Processed {video_path.name}: {len(segments)} segment(s) in {duration:.2f}s")








# ======================================== main ================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optical Flow Clipper")
    parser.add_argument("--preset", type=str, default="None", help="Preset name (e.g., TinyHigh, newPreset)")
    parser.add_argument("--overrides", type=str, default="{}", help="JSON string of overrides (e.g., '{\"MOTION_THRESHOLD\": 3.0}')")
    args = parser.parse_args()

    config = Config()
    
    
    # set the starting values for these specific mask knobs
    config.WATER_MASK_ENABLED   = WATER_MASK_ENABLED
    config.WATER_MASK_MODE      = WATER_MASK_MODE
    config.WATER_MASK_DILATE_PX = WATER_MASK_DILATE_PX
    config.WATER_SHIFT_X = WATER_SHIFT_X
    config.WATER_SHIFT_Y_TOP = WATER_SHIFT_Y_TOP
    config.WATER_SHIFT_Y_BOTTOM = WATER_SHIFT_Y_BOTTOM  
    config.WATER_SCALE_X  = WATER_SCALE_X      # new
    config.WATER_ANCHOR_X = WATER_ANCHOR_X     # new

    
    state = State()
    video_root = VIDEO_ROOT

    # Apply overrides (priority: CLI --overrides  >  canonical of_overrides.json  >  code defaults)
    override_dict: Dict[str, Any] = {}
    preset_name = args.preset

    raw_cli = (args.overrides or "").strip()

    # --- 1) CLI overrides -----------------------------------------------------
    if raw_cli and raw_cli not in ("{}", ""):
        try:
            override_dict = json.loads(raw_cli)
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid CLI override JSON: {e}")
            override_dict = {}

        if isinstance(override_dict, dict) and override_dict:
            print(f"✅ OVERRIDES: Using command-line overrides via --overrides ({len(override_dict)} key(s))")
            state.override_info = apply_overrides(config, override_dict, "command-line", preset_name)
        else:
            # empty dict (or non-dict) from CLI should behave like "no CLI overrides"
            override_dict = {}

    # --- 2) Canonical overrides file -----------------------------------------
    if ENABLE_OVERRIDES and not override_dict:
        if OVERRIDES_CANON_PATH.exists():
            try:
                with open(OVERRIDES_CANON_PATH, "r", encoding="utf-8-sig") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and payload:
                    print(f"✅ OVERRIDES: Using of_overrides.json file @ {OVERRIDES_CANON_PATH}")
                    state.override_info = apply_overrides(config, payload, str(OVERRIDES_CANON_PATH), preset_name="custom")
                    override_dict = payload
                else:
                    print(f"⚠️  OVERRIDES: {OVERRIDES_CANON_PATH} is empty or not a JSON object; using code defaults.")
            except Exception as e:
                print(f"⚠️  OVERRIDES: failed to load {OVERRIDES_CANON_PATH}: {e}")
        else:
            # --- 3) Breaker: no CLI + no file -> defaults ----------------------
            print("⚠️  OVERRIDES: No CLI overrides and no of_overrides.json file found.")
            print("⚠️  OVERRIDES: Using DEFAULT code settings.")
            try:
                confirm = input("Type DEFAULTS to proceed, or press Enter to abort: ").strip()
            except (KeyboardInterrupt, EOFError):
                confirm = ""
            if confirm != "DEFAULTS":
                print("🛑 Aborted (no overrides).")
                sys.exit(0)
            # mark defaults explicitly
            state.override_info = {
                "source": None,
                "preset_name": preset_name or "None",
                "values": {},
                "mode": "defaults",
                "enabled_count": 0,
                "applied_keys": [],
            }
    # Resolve input videos (VIDEO_ROOT is always a folder; choose single vs batch via INPUT_MODE)
    if INPUT_MODE == "single":
        single_path = VIDEO_ROOT / SINGLE_CLIP_NAME
        if not single_path.exists():
            print(f" Found 0 .mkv files: single clip not found: {single_path}")
            sys.exit(0)
        videos = [single_path]
    else:
        videos = list_mkvs(VIDEO_ROOT, config.MAX_FILES, config.RECURSIVE)
        if not videos:
            print(f" Found 0 .mkv files under VIDEO_ROOT: {VIDEO_ROOT}")
            sys.exit(0)

    print(f" Found {len(videos)} .mkv file{'s' if len(videos) != 1 else ''} under VIDEO_ROOT: {VIDEO_ROOT}")
    first_clip = videos[0]
    # paths0 = get_paths(str(first_clip), MODULE_NAME)
    paths0 = sp.get_paths(str(first_clip), MODULE_NAME)
    log_dir = Path(paths0["LOG_DIR"])

    state.camera_snapshot = get_axis_ptz_snapshot()
    state.requested_preset = state.camera_snapshot.get("preset", "Home")
    _ = build_preview_with_options(
        video_root=VIDEO_ROOT,
        first_clip_path=first_clip,
        log_dir=log_dir,
        default_source=PREVIEW_DEFAULT_SOURCE,
        reference_dir=PREVIEW_REFERENCE_DIR,
        reference_preset=PREVIEW_REFERENCE_PRESET,
        axis_ip=None,
        axis_creds_path=None,
        clip_frame_index="middle",
        auto_open=True,
        config=config,
        state=state,
    )

    # _print_active_filter_settings(config, state, camera_preset)
    _print_active_filter_settings(config, state, state.requested_preset)

    # --- NEW: write run_config.json BEFORE the final proceed prompt (so aborted runs still have config) ---
    try:
        log_dir0 = paths0["LOG_DIR"]
        log_dir0.mkdir(parents=True, exist_ok=True)

        cap0 = cv2.VideoCapture(str(videos[0]))
        fps0 = cap0.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap0.release()

        now0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload0 = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "video_name": videos[0].name,
            "video_path": str(videos[0]),
            "timestamp": now0,
            "clip_info": {"fps": float(fps0), "total_frames": int(total_frames0), "w": int(w0), "h": int(h0)},
            "knobs": {k: v for k, v in config.__dict__.items() if k in ALLOWED_OVERRIDES},
            "overrides": state.override_info,
            "camera_preset": state.requested_preset or "custom",
        }
        with open(log_dir0 / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(payload0, f, indent=2)
    except Exception as e:
        print(f"⚠ Could not write pre-run run_config.json: {e}")



    try:
        resp = input("Proceed with processing ALL clips? (y/N): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        resp = "n"
    if resp not in ("y", "yes"):
        print(" Aborted by user.")
        sys.exit(0)

    for idx, vp in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] Processing {vp.name} …")
        try:
            # process_clip(vp, paths0 if idx == 1 else get_paths(str(vp), MODULE_NAME), config, state)
            process_clip(vp, paths0 if idx == 1 else sp.get_paths(str(vp), MODULE_NAME), config, state)
        except Exception as e:
            print(f"    Error on {vp.name}: {e}")


if __name__ == "__main__":
    main()
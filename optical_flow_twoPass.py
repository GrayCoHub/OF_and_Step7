# optical_flow_twoPass.py


# Refactor goals:
#   - Eliminate NameError cascades (fps/state/pathlib/Path scope issues)
#   - Single source of truth for packet export (end-of-clip only)
#   - One write for flow_p2_frame_log.csv (no duplicated/indented blocks)
#   - Robust std_paths fallback that actually works if coreModules_Eagle isn't available
#   - Accurate terminal summary of where knobs came from (defaults vs CLI vs file)


'''

$ov = '{
  "MASK_GATING_PASS1": "hard",
  "MASK_GATING_PASS2": "soft",

  "MOTION_THRESHOLD": 1.9,
  "MIN_MOTION_PIXELS": 95,

  "MOTION_THRESHOLD_P2": 1.6,
  "MIN_MOTION_PIXELS_P2": 70,
  "P2_PAD_FRAMES": 6,

  "MERGE_GAP_FRAMES": 24,
  "MORPH_OPEN_ITERS": 0,
  "MORPH_CLOSE_ITERS": 1
}'

python .\optical_flow_twoPass.py --overrides $ov


The overrides.json File Location: C:\AxisRecordings\EagleDetection\outputs\overrides_file

'''


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import argparse
import csv
import glob
import json
import os
import sys
import uuid

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ======================================================================================
# ENV / WATCHER
# ======================================================================================
# Force watcher OFF unless caller explicitly set it earlier
os.environ["STD_PATHS_WATCHER"] = os.environ.get("STD_PATHS_WATCHER", "0")


# ======================================================================================
# PATHS (std_paths_of preferred, local fallback guaranteed)
# ======================================================================================

BASE_PROFILE_W = 1920
BASE_PROFILE_H = 1080


class _LocalStdPaths:
    """
    Local fallback when coreModules_Eagle.std_paths_of isn't available.

    Folder structure:
      <output_root>/<module_name>/<run_name>/{logs,frames,videos}

    Returns a dict with legacy string keys plus *_P Path keys for safer code.
    """

    def __init__(self) -> None:
        self._output_root: Optional[str] = None

    def set_output_root(self, root: str) -> None:
        self._output_root = str(root)

    def get_paths(self, video_path: str, module_name: str) -> Dict[str, Any]:
        vp = Path(video_path)
        stem = vp.stem
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{stem}_{stamp}_{uuid.uuid4().hex[:4]}"

        out_root = Path(self._output_root) if self._output_root else (Path.cwd() / "outputs")
        run_dir = out_root / module_name / run_name

        logs = run_dir / "logs"
        frames = run_dir / "frames"
        videos = run_dir / "videos"
        for d in (logs, frames, videos):
            d.mkdir(parents=True, exist_ok=True)

        return {
            "OUT_DIR": str(run_dir),
            "LOG_DIR": str(logs),
            "IMAGE_DIR": str(frames),  # historical key
            "VIDEO_DIR": str(videos),
            "RUN_NAME": run_name,
            "OUT_DIR_P": run_dir,
            "LOG_DIR_P": logs,
            "IMAGE_DIR_P": frames,
            "VIDEO_DIR_P": videos,
        }


# Prefer core std_paths_of; fall back to local version if repo isn't present
try:
    from coreModules_Eagle import std_paths_of as sp  # type: ignore
except Exception:
    sp = _LocalStdPaths()

# Output root resolution (priority):
#   env OF_OUTPUT_ROOT  >  hard-coded comparison root  >  std_paths default
_HARDCODED_OUTPUT_ROOT = r"C:\Axis_code_projects\OF_vs_Step7\outputs"
_OF_ENV_ROOT = os.getenv("OF_OUTPUT_ROOT")

try:
    sp.set_output_root(_OF_ENV_ROOT if _OF_ENV_ROOT else _HARDCODED_OUTPUT_ROOT)
except Exception:
    # if core std_paths_of doesn't expose set_output_root, ignore (it will use its internal root)
    pass


MODULE_NAME = "OF_HO"
SCRIPT_VERSION = "2.1-refac"

# define module_root ONCE, at the top of the export section 
module_root = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")



# ======================================================================================
# ======================================================================================

# ======================================================================================
# ======================================================================================
# INPUT CONFIG (FORGET-PROOF)
# ======================================================================================

VIDEO_ROOT = Path(r"C:\AxisRecordings\Optical_Flow\videos")  # always a folder
INPUT_MODE = "single"   # "single" or "batch"
SINGLE_CLIP_NAME = "1-smallObject-R2L-bankingTurn_L2R.mkv" 
VIDEO_EXTS = (".mkv",)

# "C:\AxisRecordings\Optical_Flow\videos\1-smallObject-R2L-bankingTurn_L2R.mkv"
# "big_bird_R2L.mkv"
# "C:\AxisRecordings\Optical_Flow\videos\1010-1641-2-115-small.mkv"


# ======================================================================================
# ======================================================================================

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

PREVIEW_DEFAULT_SOURCE = "clip"
PREVIEW_REFERENCE_DIR = Path(r"output/eagleEngine_mass_prod/frame_metadata")
PREVIEW_REFERENCE_PRESET = "20251014_135916_2154_20251014_224421"

DOWNLOADS_DIR = Path(r"C:\Users\prior\Downloads")
DOWNLOAD_SNAPSHOT_GLOB = "snapshot_*.jpg"

STATUS_EVERY = 25

AXIS_IP_DEFAULT = "192.168.1.146"
AXIS_CREDS_PATH_DEFAULT = r"C:\All_api_keys\axis_q6155\axis_creds.json"



#  Toggle for auto-tuner ... False will result in default config settings (when no CL or .json of used)
AUTO_TUNER_ENABLED = False        # set True or False to use

# ---------------------------------------------------------
# Operational toggles
# ---------------------------------------------------------
ENABLE_PREVIEW = False   # Set False to skip preview/prompt

# ---------------------------------------------------------
# Output / Rendering Toggles (high I/O cost)
# ---------------------------------------------------------
ENABLE_SAVE_FRAME_JPEGS      = False   # heavy I/O
ENABLE_SAVE_SEGMENT_VIDEOS   = False   # heavy I/O
ENABLE_SAVE_COMPILED_VIDEO   = False   # heavy I/O
ENABLE_BOX_OVERLAY           = False   # drawing cost





# ======================================================================================
# MASK PRESETS (water + ROI)
# ======================================================================================

WATER_MASK_ENABLED   = True
WATER_MASK_MODE      = "exclude"   # "exclude"|"include"
WATER_MASK_DILATE_PX = 15
WATER_SHIFT_X        = 0
WATER_SHIFT_Y_TOP    = -185
WATER_SHIFT_Y_BOTTOM = 5
WATER_SCALE_X        = 1.0
WATER_ANCHOR_X       = "center"    # "left"|"center"|"right"

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

CAMERA_PRESETS = {
    "Home": {
        "ptz": {"pan_deg": 0.0, "tilt_deg": 0.0, "zoom": 1.0},
        "water_profile": HOME_PROFILE_ABS,
        "rois": [],
        "rois_names": [],
    },
    "Harbor": {
        "ptz": {"pan_deg": 0.0, "tilt_deg": 8.0, "zoom": 240.0},
        "water_profile": HARBOR_PROFILE_ABS,
        "rois": [
            (200, 900, 400, 1000),
            (1200, 100, 1800, 600),
            (0, 900, 1919, 1079),
        ],
        "rois_names": ["harbor_left", "harbor_right", "grass"],
    },
}


def _water_polygon(preset_data: dict, config: "Config") -> Tuple[Optional[np.ndarray], int]:
    """Build a single polygon in BASE_PROFILE_W/H coordinates, applying shifts + x-scale."""
    profile = preset_data.get("water_profile") or []
    if not profile:
        return (None, 0)

    dx  = int(getattr(config, "WATER_SHIFT_X", 0))
    dyt = int(getattr(config, "WATER_SHIFT_Y_TOP", 0))
    dyb = int(getattr(config, "WATER_SHIFT_Y_BOTTOM", 0))
    scale_x  = float(getattr(config, "WATER_SCALE_X", 1.0))
    anchor_s = str(getattr(config, "WATER_ANCHOR_X", "center")).lower()

    if anchor_s == "left":
        ax = 0.0
    elif anchor_s == "right":
        ax = float(BASE_PROFILE_W)
    else:
        ax = float(BASE_PROFILE_W) / 2.0

    top: List[Tuple[int, int]] = []
    bot: List[Tuple[int, int]] = []
    for (x, y, z) in profile:
        xt = float(x + dx)
        yt = int(y + dyt)
        zb = int(z + dyb)

        xs = ax + (xt - ax) * scale_x
        xs_i = int(min(BASE_PROFILE_W - 1, max(0, round(xs))))
        yt   = int(min(2000, max(-2000, yt)))
        zb   = int(min(2000, max(-2000, zb)))

        top.append((xs_i, yt))
        bot.append((xs_i, zb))

    pts = top + bot[::-1]
    if len(pts) < 3:
        return (None, 0)

    return (np.asarray(pts, dtype=np.int32), len(pts))


# ======================================================================================
# CONFIG + STATE
# ======================================================================================


@dataclass
class Config:
    # Pass-1 thresholds
    MOTION_THRESHOLD: float = 2.8
    MIN_MOTION_PIXELS: int = 250

    # Pass-2 recovery thresholds
    MOTION_THRESHOLD_P2: float = 2.0
    MIN_MOTION_PIXELS_P2: int = 150
    P2_PAD_FRAMES: int = 4

    # segmentation
    MIN_SEGMENT_LENGTH: int = 6
    MERGE_GAP_FRAMES: int = 20

    # gating
    MASK_GATING_PASS1: str = "hard"   # "soft"|"hard"
    MASK_GATING_PASS2: str = "hard"   # "soft"|"hard"

    # morphology
    USE_MORPHOLOGY: bool = True
    MORPH_KERNEL_SZ: int = 3
    MORPH_OPEN_ITERS: int = 1
    MORPH_CLOSE_ITERS: int = 2

    # output / overlays
    SAVE_FRAME_JPEGS: bool = True
    SAVE_PER_SEGMENT_VIDEO: bool = True
    SAVE_COMPILED_VIDEO: bool = True
    BOX_OVERLAY: bool = True
    BOX_MIN_PIXELS: int = 200
    BOX_THICKNESS: int = 2

    # ROI / water
    ROI_FILTER_ENABLED: bool = True
    ROI_FILTER_MODE: str = "exclude"

    WATER_MASK_ENABLED: bool = True
    WATER_MASK_MODE: str = "exclude"
    WATER_MASK_DILATE_PX: int = 2
    WATER_SHIFT_X: int = 0
    WATER_SHIFT_Y_TOP: int = 0
    WATER_SHIFT_Y_BOTTOM: int = 0
    WATER_SCALE_X: float = 1.0
    WATER_ANCHOR_X: str = "center"

    # caption
    CAPTION_MARGIN: int = 12
    CAPTION_FONTSCALE: float = 0.6
    CAPTION_THICKNESS: int = 1

    # diagnostics
    LOW_DENSITY_WARN: float = 0.20

    # file discovery
    MAX_FILES: int = 999999
    RECURSIVE: bool = True
    
    # auto-tune toggle
    AUTO_TUNER_ENABLED: bool = True   # default On


@dataclass
class State:
    override_info: Dict[str, Any] = field(default_factory=dict)
    camera_snapshot: Optional[Dict[str, Any]] = None
    requested_preset: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.override_info:
            self.override_info = {
                "source": None,
                "preset_name": "None",
                "values": {},
                "mode": "defaults",
                "enabled_count": 0,
                "applied_keys": [],
            }


ALLOWED_OVERRIDES = {
    "MOTION_THRESHOLD": float,
    "MIN_MOTION_PIXELS": int,
    "MOTION_THRESHOLD_P2": float,
    "MIN_MOTION_PIXELS_P2": int,
    "P2_PAD_FRAMES": int,
    "MIN_SEGMENT_LENGTH": int,
    "MERGE_GAP_FRAMES": int,
    "MASK_GATING_PASS1": str,
    "MASK_GATING_PASS2": str,
    "USE_MORPHOLOGY": bool,
    "MORPH_KERNEL_SZ": int,
    "MORPH_OPEN_ITERS": int,
    "MORPH_CLOSE_ITERS": int,
    "BOX_OVERLAY": bool,
    "BOX_MIN_PIXELS": int,
    "ROI_FILTER_ENABLED": bool,
    "WATER_MASK_ENABLED": bool,
}

DISALLOWED_MASK_KEYS = {
    "WATER_MASK_MODE",
    "WATER_MASK_DILATE_PX",
    "WATER_SHIFT_X",
    "WATER_SHIFT_Y_TOP",
    "WATER_SHIFT_Y_BOTTOM",
    "WATER_SCALE_X",
    "WATER_ANCHOR_X",
    "ROI_FILTER_MODE",
}


OVERRIDES_CANON_PATH = Path(r"C:\AxisRecordings\EagleDetection\outputs\overrides_file\of_overrides.json")
ENABLE_OVERRIDES = True


def apply_overrides(config: Config, override_dict: Dict[str, Any], source: str, preset_name: str = "custom") -> Dict[str, Any]:
    info = {
        "source": source,
        "preset_name": preset_name,
        "values": {},
        "mode": "sparse",
        "enabled_count": 0,
        "applied_keys": [],
    }

    if not isinstance(override_dict, dict) or not override_dict:
        print(f"⚠️  Overrides: empty from {source}")
        return info

    flat = override_dict
    info["values"] = flat
    info["enabled_count"] = len(flat)
    info["applied_keys"] = list(flat.keys())

    for key, value in flat.items():
        if key in DISALLOWED_MASK_KEYS:
            print(f"⚠️  Ignoring override for '{key}' (mask/shift keys are not settable via JSON).")
            continue

        caster = ALLOWED_OVERRIDES.get(key)
        if caster is None:
            print(f"⚠️  Unknown override key: {key}")
            continue

        if key in ("MASK_GATING_PASS1", "MASK_GATING_PASS2"):
            if str(value) not in ("soft", "hard"):
                print(f"⚠️  Invalid value for {key}: {value}; must be 'soft' or 'hard'")
                continue
            setattr(config, key, str(value))
            continue

        try:
            setattr(config, key, caster(value))
        except Exception as e:
            print(f"⚠️  Invalid override for {key}: {value} ({e})")

    print(f"[OVERRIDES] preset: {preset_name} ({info['enabled_count']} key(s) from {source}: {info['applied_keys']})")
    return info


# ======================================================================================
# VIDEO LISTING
# ======================================================================================

def list_mkvs(video_root: Path, max_files: int, recursive: bool) -> List[Path]:
    pattern = "**/*.mkv" if recursive else "*.mkv"
    return sorted(video_root.glob(pattern))[:max_files]


def get_clip_info(video_path: Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0, 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 1e-6:
        fps = 25.0
    return fps, total, w, h


# ======================================================================================
# AXIS PREVIEW HELPERS
# ======================================================================================

def get_axis_ptz_snapshot(axis_ip: str = None, axis_creds_path: str = None) -> Dict[str, Any]:
    try:
        axis_ip = axis_ip or AXIS_IP_DEFAULT
        axis_creds_path = axis_creds_path or AXIS_CREDS_PATH_DEFAULT

        with open(axis_creds_path, "r", encoding="utf-8-sig") as f:
            creds = json.load(f)

        auth = HTTPDigestAuth(creds["username"], creds["password"])
        r = requests.get(f"http://{axis_ip}/axis-cgi/com/ptz.cgi?query=position", auth=auth, timeout=3)
        r.raise_for_status()

        snap: Dict[str, str] = {}
        for line in r.text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                snap[k.strip()] = v.strip()

        def _f(k, cast=float, default=None):
            try:
                return cast(snap.get(k))
            except Exception:
                return default

        # TODO: map PTZ -> preset (for now default Home)
        return {
            "preset": "Home",
            "pan_deg": _f("pan", float, 0.0),
            "tilt_deg": _f("tilt", float, 0.0),
            "zoom": _f("zoom", int, 1),
        }
    except Exception as e:
        print(f"⚠️ PTZ snapshot error: {e}")
        return {"preset": "Home", "pan_deg": 0.0, "tilt_deg": 0.0, "zoom": 1}


def _fetch_axis_snapshot(axis_ip: str, axis_creds_path: str) -> np.ndarray:
    try:
        axis_ip = axis_ip or AXIS_IP_DEFAULT
        axis_creds_path = axis_creds_path or AXIS_CREDS_PATH_DEFAULT

        with open(axis_creds_path, "r", encoding="utf-8-sig") as f:
            creds = json.load(f)

        auth = HTTPDigestAuth(creds["username"], creds["password"])
        r = requests.get(f"http://{axis_ip}/axis-cgi/jpg/image.cgi", auth=auth, timeout=3)
        r.raise_for_status()

        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None else np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ Snapshot error: {e}")
        return np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8)


def _extract_middle_frame(video_path: str) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8), 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8), idx
    return frame, idx


def get_preview_image(
    source: str,
    clip_path: Path,
    ref_dir: Path,
    ref_preset: str,
    axis_ip: str,
    axis_creds: str,
) -> Tuple[np.ndarray, str]:
    source = (source or "clip").lower().strip()

    if source == "live":
        img = _fetch_axis_snapshot(axis_ip or AXIS_IP_DEFAULT, axis_creds or AXIS_CREDS_PATH_DEFAULT)
        return img, f"live@{axis_ip or AXIS_IP_DEFAULT}"

    if source == "downloads":
        files = sorted(DOWNLOADS_DIR.glob(DOWNLOAD_SNAPSHOT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            path = files[0]
            img = cv2.imread(str(path))
            if img is None:
                print(f"⚠️  Could not read snapshot: {path}")
                img = np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8)
            return img, f"downloads:{path.name}"
        print(f"⚠️  No files matching {DOWNLOAD_SNAPSHOT_GLOB} in {DOWNLOADS_DIR}")
        return np.zeros((BASE_PROFILE_H, BASE_PROFILE_W, 3), dtype=np.uint8), "downloads:<none>"

    # reference not wired (kept simple)
    if source == "reference":
        print("⚠️  Reference preview not configured; falling back to clip middle frame.")
        img, fi = _extract_middle_frame(str(clip_path))
        return img, f"clip:{clip_path.name}@frame={fi}"

    img, fi = _extract_middle_frame(str(clip_path))
    return img, f"clip:{clip_path.name}@frame={fi}"


def draw_preview_caption(img: np.ndarray, config: Config, state: State) -> None:
    margin = int(getattr(config, "CAPTION_MARGIN", 12))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = float(getattr(config, "CAPTION_FONTSCALE", 0.6))
    th     = int(getattr(config, "CAPTION_THICKNESS", 1))

    preset = state.requested_preset or "Home"
    ptz = state.camera_snapshot or {"pan_deg": 0.0, "tilt_deg": 0.0, "zoom": 0}

    if state.camera_snapshot is None:
        row1 = f"Preset Settings: {preset}"
    else:
        row1 = f"Camera Settings: {preset} | pan={ptz.get('pan_deg')} tilt={ptz.get('tilt_deg')} zoom={ptz.get('zoom')}"

    y = margin + 16
    cv2.putText(img, row1, (margin, y), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, row1, (margin, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    ov = state.override_info or {}
    src = ov.get("source") or "defaults"
    enabled = int(ov.get("enabled_count") or 0)
    row2 = f"Knobs: {src} ({enabled} override key(s))"
    y = margin + 32
    cv2.putText(img, row2, (margin, y), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, row2, (margin, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    y = margin + 52
    preset_data = CAMERA_PRESETS.get(preset) or {}
    water_profile = preset_data.get("water_profile") or []

    # WATER overlay
    if bool(getattr(config, "WATER_MASK_ENABLED", False)) and water_profile:
        legend = f"WATER: {getattr(config, 'WATER_MASK_MODE', 'exclude')} | pts={len(water_profile)}"
        cv2.putText(img, legend, (margin, y), font, fs, (40, 255, 40), 2, cv2.LINE_AA)
        y += 20

        poly, npts = _water_polygon(preset_data, config)
        if poly is not None and npts >= 3:
            h_img, w_img = img.shape[:2]
            sx = w_img / float(BASE_PROFILE_W)
            sy = h_img / float(BASE_PROFILE_H)

            poly_scaled = poly.copy()
            poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
            poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_scaled], color=(0, 255, 0))
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0.0, dst=img)


def build_preview_with_options(
    *,
    first_clip_path: Path,
    log_dir: Path,
    default_source: str,
    reference_dir: Path,
    reference_preset: str,
    axis_ip: Optional[str],
    axis_creds_path: Optional[str],
    auto_open: bool,
    config: Config,
    state: State,
) -> Path:
    # pick source interactively
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

    # choose preset if non-live
    if source == "live":
        snap = get_axis_ptz_snapshot(axis_ip=axis_ip, axis_creds_path=axis_creds_path)
        state.camera_snapshot = snap
        state.requested_preset = (snap.get("preset") or "Home")
    else:
        state.camera_snapshot = None
        preset_keys = list(CAMERA_PRESETS.keys())
        default_preset = state.requested_preset if state.requested_preset in preset_keys else "Home"
        print(f"Available camera presets: {', '.join(preset_keys)}")
        sel = input(f"Choose preset for mask/caption [{default_preset}]: ").strip()
        state.requested_preset = sel if sel in preset_keys else default_preset

    img, desc = get_preview_image(
        source=source,
        clip_path=first_clip_path,
        ref_dir=reference_dir,
        ref_preset=reference_preset,
        axis_ip=axis_ip or AXIS_IP_DEFAULT,
        axis_creds=axis_creds_path or AXIS_CREDS_PATH_DEFAULT,
    )

    print(f"[SHIFT] X={config.WATER_SHIFT_X}  YT={config.WATER_SHIFT_Y_TOP}  YB={config.WATER_SHIFT_Y_BOTTOM}  dilate={config.WATER_MASK_DILATE_PX}")

    draw_preview_caption(img, config, state)

    safe_desc = desc.replace(":", "_")
    out_path = (Path(log_dir) / f"preview_{safe_desc}.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"👀 Preview saved → {out_path}")
    print(f"[PREVIEW] source={source} preset={state.requested_preset}")

    if auto_open:
        try:
            os.startfile(str(out_path))  # type: ignore[attr-defined]
        except Exception:
            pass

    return out_path

# GS
print("\n \t OF_SCRIPT_START Line# 712 \n")

        
        
def _print_active_filter_settings(config: Config, state: State) -> None:
    preset = state.requested_preset or "Home"
    preset_data = CAMERA_PRESETS.get(preset, {"water_profile": [], "rois": []})

    ov = state.override_info or {}
    src = ov.get("source") or "defaults"
    enabled = int(ov.get("enabled_count") or 0)

    print("\nActive Settings Summary:")
    print(f"  Camera Preset: {preset}")
    print("    => affects mask geometry, water profile, caption overlays")

    # --- OVERRIDE SOURCE ---
    print(f"  Knob Source: {src} ({enabled} override key(s))")

    # --- TUNER STATUS ---
    tuner_enabled = getattr(state, "tuner_enabled", False)
    tuner_reason  = getattr(state, "tuner_reason", "unknown")
    tuner_diff    = getattr(state, "tuner_diff", None)

    print("  Tuner Status:")
    print(f"    enabled: {tuner_enabled}")
    print(f"    reason: {tuner_reason}")

    # If tuner ran, show classification
    if tuner_diff:
        print(f"    classification: {tuner_diff.get('classification', 'unknown')}")

    # --- KEY GATING PARAMETERS ---
    print("  Some Key (custom-set) Gate Values Used Here: ")
    print(f"    MOTION_THRESHOLD: {config.MOTION_THRESHOLD}")
    print(f"    MIN_MOTION_PIXELS: {config.MIN_MOTION_PIXELS}")
    print(f"    MASK_GATING_PASS1: {config.MASK_GATING_PASS1}")
    print(f"    MASK_GATING_PASS2: {config.MASK_GATING_PASS2}")

    # --- WATER MASK ---
    if config.WATER_MASK_ENABLED:
        print(f"  WATER_MASK: {config.WATER_MASK_MODE} ({len(preset_data.get('water_profile') or [])} points)")



# ======================================================================================
# MASK BUILD
# ======================================================================================

def build_keep_mask(config: Config, frame_shape: Tuple[int, int], preset: str) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    preset_data = CAMERA_PRESETS.get(preset, {"water_profile": [], "rois": []})

    # ROI (unscaled; assumes 1920x1080 recordings — keep as-is for now)
    rois = preset_data.get("rois") or []
    if config.ROI_FILTER_ENABLED and rois:
        mode = (config.ROI_FILTER_MODE or "exclude").lower()
        if mode == "include":
            mask[:, :] = 0
            for (x1, y1, x2, y2) in rois:
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
        else:
            for (x1, y1, x2, y2) in rois:
                mask[int(y1):int(y2), int(x1):int(x2)] = 0

    # Water
    poly, npts = (None, 0)
    if config.WATER_MASK_ENABLED and (preset_data.get("water_profile") or []):
        poly, npts = _water_polygon(preset_data, config)

    if poly is not None and npts >= 3:
        sx = w / float(BASE_PROFILE_W)
        sy = h / float(BASE_PROFILE_H)
        poly_scaled = poly.copy()
        poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
        poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

        water_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(water_mask, [poly_scaled], 255)

        k = int(getattr(config, "WATER_MASK_DILATE_PX", 0) or 0)
        if k > 0:
            kernel = np.ones((k, k), np.uint8)
            water_mask = cv2.dilate(water_mask, kernel)

        if (getattr(config, "WATER_MASK_MODE", "exclude") or "exclude") == "exclude":
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(water_mask))
        else:
            mask = cv2.bitwise_and(mask, water_mask)

    return mask


# ======================================================================================
# LOGGERS
# ======================================================================================

class PerFrameLogger:
    def __init__(self, path: Path, total_frames: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        self.w.writerow(["pass","frame_idx","active_raw","active_gated","kept","flow_dx_mean","flow_speed_mean"])
        self._closed = False
        self.total_frames = int(total_frames) if total_frames is not None else None

    def log(self, pass_tag: str, frame_idx: int, active_raw: int, active_gated: int, kept: bool,
            flow_dx_mean: float, flow_speed_mean: float) -> None:
        self.w.writerow([pass_tag, int(frame_idx), int(active_raw), int(active_gated),
                         bool(kept), float(flow_dx_mean), float(flow_speed_mean)])

    def close(self) -> None:
        if self._closed:
            return
        self.fh.close()
        self._closed = True


# ======================================================================================
# SEGMENTS / DENSITY
# ======================================================================================

def _segments_from_kept_frames(kept_frames: List[int], config: Config) -> List[Tuple[int, int]]:
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
    s = int(start)
    e = int(end)
    return sum(1 for k in kept if s <= int(k) <= e)


def _segment_density(kept: set, seg: Tuple[int, int]) -> Tuple[int, int, float]:
    span_len = _segment_span_len(seg)
    md_count = _count_kept_in_span(kept, seg[0], seg[1])
    density = (md_count / span_len) if span_len > 0 else 0.0
    return int(md_count), int(span_len), float(density)


def _longest_by_span(segments: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not segments:
        return None
    return max(segments, key=lambda se: (_segment_span_len(se), -int(se[0])))
    
    
    
# ======================================================================================
# PASS-1: DETECT
# ======================================================================================

def detect_motion_frames(
    video_path: Path,
    config: Config,
    preset: str,
    logger: Optional[PerFrameLogger] = None,
) -> Tuple[List[Tuple[int, int]], set, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to open video: {video_path}")
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
            logger.log("p1", idx, active_raw, active_gated, kept, flow_dx_mean, flow_speed_mean)

        if kept:
            kept_p1.add(idx)

        prev_gray = gray

        if idx % STATUS_EVERY == 0:
            # print(f"   ▶ Frame {idx:05d}/{total_frames} | raw={active_raw} gated={active_gated} kept={1 if kept else 0}")
            print(f"   > Frame {idx:05d}/{total_frames} | raw={active_raw} gated={active_gated} kept={1 if kept else 0}")


    cap.release()

    segments = _segments_from_kept_frames(sorted(kept_p1), config)
    return segments, kept_p1, total_frames


# ======================================================================================
# PASS-2: RECOVER + REBUILD FINAL
# ======================================================================================

def p2_recover_and_rebuild_segments(
    p1_segments: List[Tuple[int, int]],
    kept_p1: set,
    video_path: Path,
    config: Config,
    preset: str,
    log_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[int, int]], set, set, Dict[str, Any], List[Tuple[Any, ...]], List[str]]:
    """
    Pass-2 recovery:
      - scans padded windows around each P1 segment using P2 thresholds
      - produces kept_p2 (frames not in kept_p1)
      - kept_final = kept_p1 ∪ kept_p2
      - rebuild segments from kept_final

    Writes flow_p2_frame_log.csv once (if log_dir provided).

    Returns:
      final_segments, kept_p2, kept_final, stats, audit_rows, audit_header
    """
    audit_header = [
        "frame_idx", "in_p1_span", "in_p2_pad",
        "md_p1", "md_p2", "md_final",
        "active_raw_p2", "flow_energy_p2", "flow_energy_norm_p2"
    ]
    audit_rows: List[Tuple[Any, ...]] = []

    if not p1_segments:
        stats = {
            "p1_segments": 0,
            "p2_windows": 0,
            "p1_frames": int(len(kept_p1)),
            "p2_recovered_frames": 0,
            "p2_recovered_in_pad": 0,
            "final_frames": int(len(kept_p1)),
            "final_segments": 0,
            "gain_frames": 0,
        }
        return [], set(), set(kept_p1), stats, audit_rows, audit_header

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to open video for P2 recovery: {video_path}")
        return p1_segments, set(), set(kept_p1), {}, audit_rows, audit_header

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

    pad = int(getattr(config, "P2_PAD_FRAMES", 0) or 0)
    thr2 = float(getattr(config, "MOTION_THRESHOLD_P2", config.MOTION_THRESHOLD))
    pix2 = int(getattr(config, "MIN_MOTION_PIXELS_P2", config.MIN_MOTION_PIXELS))

    kept_p2: set = set()
    recovered_in_pad = 0

    def _clamp(a: int) -> int:
        return max(0, min(int(a), total_frames - 1))

    for (s, e) in p1_segments:
        win_s = _clamp(s - pad)
        win_e = _clamp(e + pad)
        if win_e <= win_s:
            continue

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
            kept = (active_gated > pix2)

            if active_gated > 0:
                flow_energy_p2 = float(mag[motion_mask > 0].sum())
                flow_energy_norm_p2 = float(flow_energy_p2 / max(active_gated, 1))
            else:
                flow_energy_p2 = 0.0
                flow_energy_norm_p2 = 0.0

            in_p1_span = int(s <= idx <= e)
            in_pad = int((idx < s) or (idx > e))

            if kept and (idx not in kept_p1):
                kept_p2.add(idx)
                if in_pad:
                    recovered_in_pad += 1

            audit_rows.append((
                idx, in_p1_span, in_pad,
                int(idx in kept_p1),
                int(kept),
                int((idx in kept_p1) or kept),
                active_raw,
                flow_energy_p2,
                flow_energy_norm_p2
            ))

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

    # Write audit CSV once (NO packet export here; packet export is centralized at end of process_clip)
    if log_dir is not None:
        try:
            out = Path(log_dir) / "flow_p2_frame_log.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(audit_header)
                w.writerows(audit_rows)
        except Exception as e:
            print(f"   ⚠️ failed to write P2 recovery audit CSV: {e}")

    print(
        f"   OK... P2 recovery: +{stats['gain_frames']} frame(s) "
        f"(P2-only={stats['p2_recovered_frames']}, in_pad={stats['p2_recovered_in_pad']}); "
        f"final_segments={stats['final_segments']}"
    )
    return final_segments, kept_p2, kept_final, stats, audit_rows, audit_header


# ======================================================================================
# RUN OUTPUTS
# ======================================================================================

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
    log_dir.mkdir(parents=True, exist_ok=True)

    seg_log = log_dir / "flow_segment_log.csv"
    master = log_dir / "flow_segment_log_master.csv"
    overrides_log = log_dir / "overrides_log.txt"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [(i, s, e, (e - s + 1) / max(fps, 1e-6)) for i, (s, e) in enumerate(segments, 1)]

    with seg_log.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "end_frame", "duration_sec"])
        w.writerows(rows)

    with master.open("a", newline="", encoding="utf-8") as f:
        f.write(f"# Run at {now}\n")
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "end_frame", "duration_sec"])
        w.writerows(rows)
        f.write(f"# Processing time: {duration:.2f} seconds\n\n")

    with overrides_log.open("a", encoding="utf-8") as f:
        f.write(f"Run at {now}\n")
        f.write(f"Preset: {(state.override_info or {}).get('preset_name', 'None')}\n")
        f.write(f"Overrides: {json.dumps((state.override_info or {}).get('values', {}), indent=2)}\n\n")

        # --- ALWAYS WRITE TUNER STATUS ---
        tuner_enabled = getattr(state, "tuner_enabled", False)
        tuner_reason  = getattr(state, "tuner_reason", "unknown")

        f.write(f"Tuner Enabled: {tuner_enabled}\n")
        f.write(f"Tuner Reason: {tuner_reason}\n")

        # --- ONLY WRITE DIFF WHEN TUNER RAN ---
        tuner_diff = getattr(state, "tuner_diff", None)
        if tuner_diff:
            f.write("=== AUTO-TUNER OVERRIDES ===\n")
            f.write(f"classification: {tuner_diff['classification']}\n")
            f.write("changes:\n")
            for key in tuner_diff["before"].keys():
                b = tuner_diff["before"][key]
                a = tuner_diff["after"][key]
                f.write(f"  {key}: {b} → {a}\n")
        else:
            f.write("No tuner overrides applied.\n")

        f.write("\n")

       
    payload = {
    "module": MODULE_NAME,
    "script_version": SCRIPT_VERSION,
    "video_name": video_path.name,
    "video_path": str(video_path),
    "timestamp": now,
    "clip_info": {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "w": int(frame_w),
        "h": int(frame_h)
    },

    "knobs": {
        k: getattr(config, k)
        for k in ALLOWED_OVERRIDES.keys()
        if hasattr(config, k)
    },

    "overrides": state.override_info,

    # --- NEW FIELD ---
    "tuner_enabled": getattr(state, "tuner_enabled", False),

    "camera_preset": state.requested_preset or "custom",
    }

    
    (log_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")



# GS
print("\n \t OF_SCRIPT_MID Line# 1212 \n")



# GS
# ============================================================
#  PRESCAN MODULE
#  ------------------------------------------------------------
#  This function runs BEFORE run_of_pass1() and BEFORE any
#  config-dependent operations. It inspects the first N frames
#  of the video, computes motion footprint metrics, and returns
#  a dictionary of prescan metrics for classification.
#
#  Integration point:
#      AFTER:  video = load_video(...)
#      BEFORE: p1_results = run_of_pass1(...)
# ============================================================



def prescan(cap, config, preset, num_frames=40, low_flow_thresh=0.5):
    """
    Perform a lightweight optical-flow prescan on the first N frames.

    Parameters
    ----------
    video : VideoReader-like object
        Must support iteration or random access to frames.
    num_frames : int
        Number of frames to analyze for prescan.
    low_flow_thresh : float
        Threshold for counting "active" pixels.

    Returns
    -------
    dict
        A dictionary of prescan metrics used by classify_target().
    """

    # ------------------------------------------------------------
    # Initialize accumulators for all prescan metrics
    # ------------------------------------------------------------
    flow_mags = []              # per-frame p95 flow magnitude
    active_pixel_counts = []    # per-frame count of active pixels
    bboxes = []                 # per-frame bounding box areas
    activation_flags = []       # 1 if frame has any motion, else 0
    centroids = []              # for centroid jitter
    flow_energy_vals = []       # normalized flow energy

    # ------------------------------------------------------------
    # Iterate through the first N frames (or until video ends)
    # ------------------------------------------------------------
    prev_gray = None
    frame_count = 0

    while frame_count < num_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # compute flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

            # p95 magnitude
            p95_mag = np.percentile(mag, 95)
            flow_mags.append(p95_mag)
            
            
            # 1. Apply water mask
            if config.WATER_MASK_ENABLED:
                keep_mask = build_keep_mask(config, frame.shape[:2], preset)
                mag = mag * (keep_mask == 255)

            # 2. Threshold masked magnitude
            active_mask = mag > low_flow_thresh

            # 3. Apply ROI filtering (already included in keep_mask)
            if config.ROI_FILTER_ENABLED:
                active_mask = active_mask & (keep_mask == 255)

            active_count = int(active_mask.sum())
            active_pixel_counts.append(active_count)





            # bounding box
            if active_count > 0:
                ys, xs = np.where(active_mask)
                bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            else:
                bbox_area = 0
            bboxes.append(bbox_area)

            # activation flag
            activation_flags.append(1 if active_count > 0 else 0)

            # centroid jitter
            if active_count > 0:
                cx = xs.mean()
                cy = ys.mean()
                centroids.append((cx, cy))
            else:
                centroids.append(None)

            # flow energy
            flow_energy_vals.append(float(mag.mean()))

        prev_gray = gray
        frame_count += 1

    metrics = {
        "p95_flow_mag": np.percentile(flow_mags, 95) if flow_mags else 0,
        "p95_active_pixels": np.percentile(active_pixel_counts, 95) if active_pixel_counts else 0,
        "bbox_area_p95": np.percentile(bboxes, 95) if bboxes else 0,
        "activation_ratio": np.mean(activation_flags) if activation_flags else 0,
        "centroid_jitter": compute_centroid_jitter(centroids),
        "flow_energy_p95": np.percentile(flow_energy_vals, 95) if flow_energy_vals else 0,
    }

    return metrics


# ============================================================
# Helper: centroid jitter
# ============================================================


def compute_centroid_jitter(centroids):
    """
    Compute frame-to-frame centroid jitter.
    """
    pts = [c for c in centroids if c is not None]
    if len(pts) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        diffs.append(np.sqrt(dx*dx + dy*dy))

    return float(np.mean(diffs))


# GS end =====================================================================================






# ======================================================================================
# PACKET EXPORT (OF P1 vs OF FINAL)
# ======================================================================================

def _read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _md_set_from_rows(rows: list, key: str) -> set:
    out = set()
    for r in rows:
        if _safe_int(r.get(key, 0), 0) == 1:
            out.add(_safe_int(r.get("frame_idx", 0), 0))
    return out

def _segment_metrics(md_frames: set, start: int, end: int) -> dict:
    if end < start:
        start, end = end, start
    span = (end - start + 1)
    if span <= 0:
        return {"start": start, "end": end, "span": 0, "md": 0, "density": 0.0,
                "gap_frames": 0, "gap_sequences": 0, "longest_gap": 0}

    md_count = 0
    gap_frames = 0
    gap_sequences = 0
    longest_gap = 0
    in_gap = False
    cur_gap = 0

    for fi in range(start, end + 1):
        if fi in md_frames:
            md_count += 1
            if in_gap:
                gap_sequences += 1
                longest_gap = max(longest_gap, cur_gap)
                in_gap = False
                cur_gap = 0
        else:
            gap_frames += 1
            in_gap = True
            cur_gap += 1

    if in_gap:
        gap_sequences += 1
        longest_gap = max(longest_gap, cur_gap)

    dens = (md_count / span) if span else 0.0
    return {
        "start": start,
        "end": end,
        "span": span,
        "md": md_count,
        "density": dens,
        "gap_frames": gap_frames,
        "gap_sequences": gap_sequences,
        "longest_gap": longest_gap,
    }



def _write_segment_summary(packet_dir: Path, segments: list, md_frames: set) -> dict:
    """
    Write segment_summary.csv for a packet directory.

    segments: list of (start_frame, end_frame)
    md_frames: set of frame indices where MD was detected

    Returns:
        {
            "segments": [row_dicts...],
            "primary": row_dict or None
        }
    """
    packet_dir.mkdir(parents=True, exist_ok=True)
    out_csv = packet_dir / "segment_summary.csv"

    seg_rows = []

    # ------------------------------------------------------------
    # Build one row per segment with correct integer casting
    # ------------------------------------------------------------
    for seg_id, (start, end) in enumerate(segments, start=1):

        start = int(start)
        end = int(end)
        span_len = int(end - start + 1)

        # Count MD frames inside this segment
        md_count = sum(1 for f in md_frames if start <= f <= end)
        density = (md_count / span_len) if span_len > 0 else 0.0

        # Compute gap metrics
        gap_frames = 0
        gap_sequences = 0
        longest_gap = 0

        in_gap = False
        current_gap = 0

        for f in range(start, end + 1):
            if f not in md_frames:
                current_gap += 1
                if not in_gap:
                    in_gap = True
                    gap_sequences += 1
            else:
                if in_gap:
                    in_gap = False
                    gap_frames += current_gap
                    longest_gap = max(longest_gap, current_gap)
                    current_gap = 0

        # If segment ends in a gap, close it
        if in_gap:
            gap_frames += current_gap
            longest_gap = max(longest_gap, current_gap)

        # --- PATCH: enforce integer types for orchestrator compatibility ---
        row = {
            "segment_id": int(seg_id),
            "start_frame": int(start),
            "end_frame": int(end),
            "span": int(span_len),
            "md": int(md_count),
            "density": float(density),  # density stays float
            "gap_frames": int(gap_frames),
            "gap_sequences": int(gap_sequences),
            "longest_gap": int(longest_gap),
        }
        # --- END PATCH ---

        seg_rows.append(row)

    # ------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id","start_frame","end_frame","span",
                "md","density","gap_frames","gap_sequences","longest_gap"
            ]
        )
        writer.writeheader()
        for r in seg_rows:
            writer.writerow(r)

    # ------------------------------------------------------------
    # Determine primary segment (same logic as before)
    # ------------------------------------------------------------
    primary = None
    if seg_rows:
        primary = max(
            seg_rows,
            key=lambda r: (
                int(r["md"]),
                float(r["density"]),
                int(r["span"]),
                -int(r["start_frame"])
            )
        )

    return {"segments": seg_rows, "primary": primary}




def _write_frame_signal(packet_dir: Path, rows: list, md_key: str) -> dict:
    packet_dir.mkdir(parents=True, exist_ok=True)
    out_csv = packet_dir / "frame_signal.csv"

    fieldnames = [
        "frame_idx",
        "md",
        "in_p1_span",
        "in_p2_pad",
        "active_raw_p2",
        "flow_energy_p2",
        "flow_energy_norm_p2",
    ]
    md_total = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            fi = _safe_int(r.get("frame_idx", 0), 0)
            md = 1 if _safe_int(r.get(md_key, 0), 0) == 1 else 0
            md_total += md

            # --- PATCH: enforce integer types for orchestrator compatibility ---
            w.writerow({
                "frame_idx": int(fi),
                "md": int(md),
                "in_p1_span": int(_safe_int(r.get("in_p1_span", 0), 0)),
                "in_p2_pad": int(_safe_int(r.get("in_p2_pad", 0), 0)),
                "active_raw_p2": int(_safe_int(r.get("active_raw_p2", 0), 0)),
                "flow_energy_p2": float(_safe_float(r.get("flow_energy_p2", 0.0), 0.0)),
                "flow_energy_norm_p2": float(_safe_float(r.get("flow_energy_norm_p2", 0.0), 0.0)),
            })
            # --- END PATCH ---


    return {"md_total": md_total, "frame_signal_csv": str(out_csv)}

def _write_run_manifest(
    packet_dir: Path,
    *,
    module: str,
    variant: str,
    video_path: str,
    timestamp: str,
    clip_info: dict,
    knobs: dict,
    overrides: dict,
    seg_summary: dict,
    md_total: int,
) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    out_json = packet_dir / "run_manifest.json"

    primary = seg_summary.get("primary") or {}
    primary_span = [int(primary.get("start_frame", 0)), int(primary.get("end_frame", 0))]
    payload = {
        "module": module,
        "variant": variant,  # "p1" or "final"
        "video_name": Path(video_path).name,
        "video_path": str(video_path),
        "timestamp": timestamp,
        "clip_info": clip_info,
        "knobs": knobs,
        "overrides": overrides,
        "primary_span": primary_span,
        "span": int(primary.get("span", 0) or 0),
        "md_total": int(md_total),
        "primary_density": float(primary.get("density", 0.0) or 0.0),
        "segments": seg_summary.get("segments", []),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_latest_pointer_files(
    module_root: Path,
    *,
    pointer_base: str,
    packet_dir: Path,
    run_dir: Path,
    video_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write BOTH json + txt pointer files for backward/forward compatibility."""
    try:
        module_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't make dirs, just bail silently.
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "timestamp": now,
        "pointer_base": pointer_base,
        "packet_dir": str(packet_dir),
        "run_dir": str(run_dir),
        "video_name": getattr(video_path, "name", None) or str(video_path),
        "video_path": str(video_path),
    }
    if extra:
        payload.update(extra)

    # json pointer
    try:
        (module_root / f"{pointer_base}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    # legacy txt pointer: just the path, first line
    try:
        (module_root / f"{pointer_base}.txt").write_text(str(packet_dir), encoding="utf-8")
    except Exception:
        pass


def _write_latest_run_dir_pointers(
    module_root: Path,
    *,
    run_dir: Path,
    video_path: Path,
) -> None:
    """Also record latest run dir at the module root."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "timestamp": now,
        "run_dir": str(run_dir),
        "video_name": getattr(video_path, "name", None) or str(video_path),
        "video_path": str(video_path),
    }
    try:
        (module_root / "_latest_run_dir.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        (module_root / "_latest_run_dir.txt").write_text(str(run_dir), encoding="utf-8")
    except Exception:
        pass


 
def export_comparison_packets_of(
    *,
    run_dir: Path,
    log_dir: Path,
    video_path: str,
    fps: float,
    total_frames: int,
    frame_w: int,
    frame_h: int,
    p1_segments: list,
    final_segments: list,
    audit_rows: Optional[List[Tuple[Any, ...]]] = None,
    audit_header: Optional[List[str]] = None,
) -> None:
    """
    Export two OF comparison packets:
      - packet_of_p1
      - packet_of_final

    Each packet is written INSIDE the run_dir (to mirror Step7), e.g.:

      <run_dir>/
        video/
        images/
        logs/
        packet_of_p1/
        packet_of_final/

    And we also write "latest" pointer files at a fixed module root:

      C:\\Axis_code_projects\\OF_vs_Step7\\outputs\\OF_HO\\
        _latest_packet_of_p1.json / .txt
        _latest_packet_of_final.json / .txt
        _latest_run_dir.json / .txt

    Packet contents (per variant) are:

      frame_signal.csv     (NEW for OF, to match Step7)
      segment_summary.csv
      run_manifest.json
    """

    # Normalize paths
    run_dir = Path(run_dir)
    log_dir = Path(log_dir)

    # ---------------------------------------------------------------------
    # 1. Fixed module root (for "latest" pointer files)
    # ---------------------------------------------------------------------
    module_root = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")
    module_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 2. Packet directories (live INSIDE the run_dir, like Step7)
    # ---------------------------------------------------------------------
    p1_dir = run_dir / "packet_of_p1"
    final_dir = run_dir / "packet_of_final"

    # ---------------------------------------------------------------------
    # 3. Build rows_dict (frame-wise log) from either:
    #       - audit_rows/audit_header (if provided by caller), OR
    #       - flow_p2_frame_log.csv under log_dir
    #
    #    rows_dict is ALWAYS defined (possibly empty), so later code
    #    can safely reference it without "referenced before assignment".
    # ---------------------------------------------------------------------
    rows_dict: List[Dict[str, Any]] = []

    if audit_rows is not None and audit_header is not None and len(audit_header) > 0:
        # Convert tuple rows + header into list-of-dicts
        col_map = {name: i for i, name in enumerate(audit_header)}
        for r in audit_rows:
            d: Dict[str, Any] = {}
            for name, idx in col_map.items():
                d[name] = r[idx]
            rows_dict.append(d)
    else:
        # Fallback: read from the P2 log, if it exists
        p2_log = log_dir / "flow_p2_frame_log.csv"
        if p2_log.exists():
            rows_dict = _read_csv_rows(p2_log)
        else:
            # No audit rows and no P2 log: rows_dict stays empty.
            # This corresponds to "no MD info available" for OF.
            rows_dict = []
    
    print("=== OF MD DIAGNOSTIC ===")
    print(f"  rows_dict len      : {len(rows_dict)}")
    if rows_dict:
        sample = rows_dict[0]
        print(f"  sample keys        : {list(sample.keys())}")
        print(f"  sample frame_idx   : {sample.get('frame_idx')}")
        print(f"  sample md_p1       : {sample.get('md_p1')}")
        print(f"  sample md_final    : {sample.get('md_final')}")
    else:
        print("  rows_dict is EMPTY — no per-frame MD available")
    print("========================")

          

    # ---------------------------------------------------------------------
    # 4. Load run_config (if present) for metadata / knobs / overrides
    # ---------------------------------------------------------------------
    run_cfg_path = log_dir / "run_config.json"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    clip_info: Dict[str, Any] = {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "w": int(frame_w),
        "h": int(frame_h),
    }
    knobs: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    module = MODULE_NAME  # default module name

    if run_cfg_path.exists():
        try:
            rc = json.loads(run_cfg_path.read_text(encoding="utf-8"))
            now = rc.get("timestamp", now)
            module = rc.get("module", module)
            clip_info = rc.get("clip_info", clip_info) or clip_info
            knobs = rc.get("knobs", {}) or {}
            overrides = rc.get("overrides", {}) or {}
        except Exception:
            # If run_config is malformed, fall back to defaults above
            pass

    
    
    # ---------------------------------------------------------------------
    # 5. Build packet_of_p1
    # ---------------------------------------------------------------------
    md_frames_p1: Set[int] = _md_set_from_rows(rows_dict, "md_p1") if rows_dict else set()

    # ALWAYS write frame_signal.csv, even if rows_dict is empty.
    # _write_frame_signal will then produce a header-only file and md_total=0.
    fs_p1 = _write_frame_signal(p1_dir, rows_dict, md_key="md_p1")
    md_total_p1 = int(fs_p1["md_total"])

    seg_p1 = _write_segment_summary(p1_dir, p1_segments, md_frames_p1)

    _write_run_manifest(
        p1_dir,
        module=module,
        variant="p1",
        video_path=video_path,
        timestamp=now,
        clip_info=clip_info,
        knobs=knobs,
        overrides=overrides,
        seg_summary=seg_p1,
        md_total=md_total_p1,
    )

        
        
    # ---------------------------------------------------------------------
    # 6. Build packet_of_final
    #
    #    Same logic as P1, but using the "md_final" column and
    #    final_segments list.
    # ---------------------------------------------------------------------
    md_frames_final: Set[int] = _md_set_from_rows(rows_dict, "md_final") if rows_dict else set()
    md_total_final = 0

    if rows_dict:
        fs_final = _write_frame_signal(final_dir, rows_dict, md_key="md_final")
        md_total_final = int(fs_final["md_total"])
    else:
        md_total_final = len(md_frames_final)

    seg_final = _write_segment_summary(final_dir, final_segments, md_frames_final)

    _write_run_manifest(
        final_dir,
        module=module,
        variant="final",
        video_path=video_path,
        timestamp=now,
        clip_info=clip_info,
        knobs=knobs,
        overrides=overrides,
        seg_summary=seg_final,
        md_total=md_total_final,
    )

    # ---------------------------------------------------------------------
    # 7. Console summary for operator
    # ---------------------------------------------------------------------
    print(
        "   OK ... OF packets written:\n"
        f"      - {p1_dir}  (variant=p1   md_total={md_total_p1})\n"
        f"      - {final_dir} (variant=final md_total={md_total_final})"
    )

    # ---------------------------------------------------------------------
    # 8. "Latest" pointer files at module_root
    #
    #    These are used by the orchestrator to resolve:
    #      - last run_dir
    #      - last packet_of_p1 / packet_of_final
    # ---------------------------------------------------------------------
    try:
        _write_latest_run_dir_pointers(
            module_root,
            run_dir=run_dir,
            video_path=Path(video_path),
        )

        _write_latest_pointer_files(
            module_root,
            pointer_base="_latest_packet_of_p1",
            packet_dir=p1_dir,
            run_dir=run_dir,
            video_path=Path(video_path),
            extra={"variant": "p1"},
        )

        _write_latest_pointer_files(
            module_root,
            pointer_base="_latest_packet_of_final",
            packet_dir=final_dir,
            run_dir=run_dir,
            video_path=Path(video_path),
            extra={"variant": "final"},
        )

    except Exception as _e_ptr:
        # Pointer file failures should never abort export; log and continue.
        print(f"   ⚠ could not write latest pointers: {_e_ptr}")

def compute_centroid_jitter(centroids):
    pts = [c for c in centroids if c is not None]
    if len(pts) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        diffs.append((dx*dx + dy*dy) ** 0.5)

    return float(sum(diffs) / len(diffs))


# def classify_target(metrics):
    # px = metrics["p95_active_pixels"]
    # bbox = metrics["bbox_area_p95"]
    # act = metrics["activation_ratio"]

    ## Very small object
    # if px < 300 and bbox < 5000:
        # return "SMALL"

    ## Large object
    # if px > 2000 or bbox > 20000:
        # return "LARGE"

    # return "MEDIUM"

def classify_target(metrics):
    bbox = metrics["bbox_area_p95"]
    jitter = metrics["centroid_jitter"]
    energy = metrics["flow_energy_p95"]

    # SMALL OBJECT: compact footprint, low jitter, low energy
    if bbox < 8000 and jitter < 4.0 and energy < 1.5:
        return "SMALL"

    # LARGE OBJECT: wide footprint, high jitter, high energy
    if bbox > 20000 or jitter > 10.0 or energy > 3.0:
        return "LARGE"

    return "MEDIUM"


def auto_tuner(config, classification, metrics):
    """
    Adjust config parameters based on prescan classification.
    Emits a terminal diff showing exactly which knobs were changed.
    """

    # --- 1. Snapshot BEFORE values (only knobs the tuner may modify) ---
    before = {
        "MOTION_THRESHOLD": config.MOTION_THRESHOLD,
        "MIN_MOTION_PIXELS": config.MIN_MOTION_PIXELS,
        "MASK_GATING_PASS1": config.MASK_GATING_PASS1,
        "MERGE_GAP_FRAMES": config.MERGE_GAP_FRAMES,
        "P2_PAD_FRAMES": config.P2_PAD_FRAMES,
    }

    # --- 2. Apply tuner profile (mutates config) ---
    if classification == "SMALL":
        config.MOTION_THRESHOLD = 1.2
        config.MIN_MOTION_PIXELS = 20
        config.MASK_GATING_PASS1 = "soft"
        config.MERGE_GAP_FRAMES = 8
        config.P2_PAD_FRAMES = 6

    elif classification == "MEDIUM":
        config.MOTION_THRESHOLD = 1.6
        config.MIN_MOTION_PIXELS = 60
        config.MASK_GATING_PASS1 = "soft"
        config.MERGE_GAP_FRAMES = 6
        config.P2_PAD_FRAMES = 4

    elif classification == "LARGE":
        config.MOTION_THRESHOLD = 1.8
        config.MIN_MOTION_PIXELS = 90
        config.MASK_GATING_PASS1 = "hard"
        config.MERGE_GAP_FRAMES = 4
        config.P2_PAD_FRAMES = 2

    # --- 3. Snapshot AFTER values ---
    after = {
        "MOTION_THRESHOLD": config.MOTION_THRESHOLD,
        "MIN_MOTION_PIXELS": config.MIN_MOTION_PIXELS,
        "MASK_GATING_PASS1": config.MASK_GATING_PASS1,
        "MERGE_GAP_FRAMES": config.MERGE_GAP_FRAMES,
        "P2_PAD_FRAMES": config.P2_PAD_FRAMES,
    }

    # --- 4. Print terminal diff block ---
    print(f"   [TUNER] classification={classification}")
    print(f"   [TUNER] applied overrides:")

    for key in before.keys():
        b = before[key]
        a = after[key]
        print(f"      {key}: {b} → {a}")

    #return config
    # Build diff dict for logging
    tuner_diff = {
        "classification": classification,
        "before": before,
        "after": after,
    }

    return config, tuner_diff



def process_clip(video_path: Path, paths: Dict[str, Any], config: Config, state: State) -> None:
    start_time = datetime.now()

    # Normalize paths
    log_dir = Path(paths.get("LOG_DIR_P") or paths["LOG_DIR"])
    img_dir = Path(paths.get("IMAGE_DIR_P") or paths["IMAGE_DIR"])
    vid_dir = Path(paths.get("VIDEO_DIR_P") or paths["VIDEO_DIR"])

    log_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    fps, total_frames, frame_w, frame_h = get_clip_info(video_path)

    # per-frame logger
    pfl = None
    try:
        pfl = PerFrameLogger(log_dir / "flow_per_frame_log.csv", total_frames=total_frames)
    except Exception as e:
        print(f"   ⚠️ per-frame logger init failed: {e}")

    preset = state.requested_preset or "Home"

    
    

    # Only run auto-tuner if no CLI/JSON overrides were used
    
        
    # ============================================================
    # PRESCAN + CLASSIFICATION + AUTO-TUNER INSERTION POINT
    # ============================================================

    # Explicit toggle
    if not getattr(config, "AUTO_TUNER_ENABLED", True):
        print("   [TUNER] disabled by config toggle")
        state.tuner_enabled = False
        state.tuner_reason = "disabled by config toggle # 2090"
        state.tuner_diff = None

    else:
        # Determine whether any *real* overrides were used
        ov = getattr(config, "overrides", None)
        enabled = getattr(ov, "enabled_count", 0) if ov else 0
        src = ov.get("source") if ov else None
        preset_name = ov.get("preset_name") if ov else None

        has_real_overrides = (
            enabled > 0 and
            src in ("json", "cli") and
            preset_name not in (None, "None")
        )

        if has_real_overrides:
            print("   [TUNER] disabled due to real overrides")
            state.tuner_enabled = False
            state.tuner_reason = "disabled due to overrides"
            state.tuner_diff = None

        else:
            # Tuner is allowed to run
            try:
                cap_ps = cv2.VideoCapture(str(video_path))
                if cap_ps.isOpened():
                    metrics = prescan(cap_ps, config, preset, num_frames=40)
                    classification = classify_target(metrics)

                    # --- IMPORTANT: capture tuner diff ---
                    config, tuner_diff = auto_tuner(config, classification, metrics)

                    state.tuner_enabled = True
                    state.tuner_reason = "enabled"
                    state.tuner_diff = tuner_diff

                cap_ps.release()

            except Exception as e:
                print(f"   ⚠️ prescan failed (continuing with original config): {e}")
                state.tuner_enabled = False
                state.tuner_reason = "prescan failure"
                state.tuner_diff = None

    
    
    # Pass-1 + Pass-2
    p1_segments, kept_p1, _ = detect_motion_frames(video_path, config, preset, logger=pfl)
    final_segments, kept_p2, kept_final, p2_stats, audit_rows, audit_header = p2_recover_and_rebuild_segments(
        p1_segments, kept_p1, video_path, config, preset, log_dir=log_dir
    )

    # terminal summary (truth)
    p1_longest = _longest_by_span(p1_segments)
    final_longest = _longest_by_span(final_segments)
    p1_total_md = int(len(kept_p1))
    final_total_md = int(len(kept_final))

    if p1_longest is not None:
        p1_md, p1_span, p1_den = _segment_density(kept_p1, p1_longest)
        print(f"   🧩 P1:    n={len(p1_segments)} | longest=({p1_longest[0]}–{p1_longest[1]}) span={p1_span} | md={p1_md} | dens={p1_den:.1%} | md_total={p1_total_md}")
    else:
        print(f"   🧩 P1:    n=0 | md_total={p1_total_md}")

    if final_longest is not None:
        f_md, f_span, f_den = _segment_density(kept_final, final_longest)
        print(f"   🧩 FINAL: n={len(final_segments)} | longest=({final_longest[0]}–{final_longest[1]}) span={f_span} | md={f_md} | dens={f_den:.1%} | md_total={final_total_md}")
    else:
        print(f"   🧩 FINAL: n=0 | md_total={final_total_md}")

    # Writers (compiled + per-segment)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to reopen video: {video_path}")
        return

    compiled_writer = None
    if config.SAVE_COMPILED_VIDEO:
        compiled_path = vid_dir / "compiled_flow_video.mp4"
        compiled_writer = cv2.VideoWriter(str(compiled_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    if final_segments:
        segment_dir = vid_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)

        keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

        for i, (start, end) in enumerate(final_segments, 1):
            seg_writer = None
            if config.SAVE_PER_SEGMENT_VIDEO:
                seg_path = segment_dir / f"segment_{i}_{video_path.stem}.mp4"
                seg_writer = cv2.VideoWriter(str(seg_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, frame = cap.read()
            if not ok or frame is None:
                if seg_writer is not None:
                    seg_writer.release()
                continue

            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if config.SAVE_FRAME_JPEGS:
                cv2.imwrite(str(img_dir / f"flow_frame_{start:05d}.jpg"), frame)
            if compiled_writer is not None:
                compiled_writer.write(frame)
            if seg_writer is not None:
                seg_writer.write(frame)

            for fidx in range(start + 1, end + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mask = (mag > float(config.MOTION_THRESHOLD)).astype(np.uint8) * 255

                if config.MASK_GATING_PASS2 == "hard":
                    motion_mask = cv2.bitwise_and(motion_mask, keep_mask)

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

    cap.release()
    if compiled_writer is not None:
        compiled_writer.release()

    duration = (datetime.now() - start_time).total_seconds()
    save_run_outputs(log_dir, config, state, video_path, final_segments, duration, fps, total_frames, frame_w, frame_h)

    if pfl is not None:
        try:
            pfl.close()
        except Exception as e:
            print(f"   ⚠️ per-frame logger close error: {e}")

    # Centralized packet export (NO state leakage, NO NameError)
    try:
        export_comparison_packets_of(
            run_dir=log_dir.parent,
            log_dir=log_dir,
            video_path=str(video_path),
            fps=float(fps),
            total_frames=int(total_frames),
            frame_w=int(frame_w),
            frame_h=int(frame_h),
            p1_segments=p1_segments,
            final_segments=final_segments,
            audit_rows=audit_rows,
            audit_header=audit_header,
        )
    except Exception as e:
        print(f"   ⚠️ packet export failed: {e}")

    # ---------------------------------------------------------
    # INSERT NEW ML SCORE OUTPUT HERE
    # ---------------------------------------------------------

    # Score = density-weighted span of the longest FINAL segment
    # (rewards long + continuous motion coverage)
    if final_longest:
        f_md, f_span, f_den = _segment_density(kept_final, final_longest)
        score = float(f_span) * float(f_den)
    else:
        score = 0.0

    print(
        f"FINAL SCORE: {score} | TH={config.MOTION_THRESHOLD} "
        f"PIX={config.MIN_MOTION_PIXELS} GAP={config.MERGE_GAP_FRAMES}"
    )

    # ---------------------------------------------------------


    print(f"    Processed {video_path.name}: {len(final_segments)} segment(s) in {duration:.2f}s")


# GS
print("\n \t OF_SCRIPT_B4 Before main() Line# 2263 \n")

# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Optical Flow Clipper (P2 refactor)")
    parser.add_argument("--preset", type=str, default="None")
    parser.add_argument("--overrides", type=str, default="{}")
    parser.add_argument("--auto-yes", action="store_true", help="Skip interactive prompts")

    args = parser.parse_args()   # ✅ must be first

    # ✅ handle auto-yes vs prompt
    if args.auto_yes:
        resp = "y"
    else:
        try:
            resp = input("Proceed with processing ALL clips? (y/N): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            resp = "n"

    if resp not in ("y", "yes"):
        print("Aborted by user.")
        sys.exit(0)

    config = Config()
    config.AUTO_TUNER_ENABLED = AUTO_TUNER_ENABLED

    # Initialize mask knobs from top-of-file globals
    config.WATER_MASK_ENABLED   = WATER_MASK_ENABLED
    config.WATER_MASK_MODE      = WATER_MASK_MODE
    config.WATER_MASK_DILATE_PX = WATER_MASK_DILATE_PX
    config.WATER_SHIFT_X        = WATER_SHIFT_X
    config.WATER_SHIFT_Y_TOP    = WATER_SHIFT_Y_TOP
    config.WATER_SHIFT_Y_BOTTOM = WATER_SHIFT_Y_BOTTOM
    config.WATER_SCALE_X        = WATER_SCALE_X
    config.WATER_ANCHOR_X       = WATER_ANCHOR_X

    # ---------------------------------------------------------
    # Apply pipeline-level output toggles (FAST MODE)
    # ---------------------------------------------------------
    config.SAVE_FRAME_JPEGS        = ENABLE_SAVE_FRAME_JPEGS
    config.SAVE_PER_SEGMENT_VIDEO  = ENABLE_SAVE_SEGMENT_VIDEOS
    config.SAVE_COMPILED_VIDEO     = ENABLE_SAVE_COMPILED_VIDEO
    config.BOX_OVERLAY             = ENABLE_BOX_OVERLAY

    state = State()


    # Overrides priority: CLI --overrides  > canonical of_overrides.json > defaults
    override_dict: Dict[str, Any] = {}
    raw_cli = (args.overrides or "").strip()

    if raw_cli and raw_cli not in ("{}", ""):
        try:
            override_dict = json.loads(raw_cli)
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid CLI override JSON: {e}")
            override_dict = {}

        if isinstance(override_dict, dict) and override_dict:
            print(f"[OVERRIDES] Using command-line overrides via --overrides ({len(override_dict)} key(s))")
            # print(f"✅ OVERRIDES: Using command-line overrides via --overrides ({len(override_dict)} key(s))")
            state.override_info = apply_overrides(config, override_dict, "command-line", preset_name=args.preset)
        else:
            override_dict = {}

    if ENABLE_OVERRIDES and not override_dict:
        if OVERRIDES_CANON_PATH.exists():
            try:
                payload = json.loads(OVERRIDES_CANON_PATH.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict) and payload:
                    # print(f"✅ OVERRIDES: Using of_overrides.json file @ {OVERRIDES_CANON_PATH}")
                    print(f"[OVERRIDES] Using command-line overrides via --overrides ({len(override_dict)} key(s))")
                    state.override_info = apply_overrides(config, payload, str(OVERRIDES_CANON_PATH), preset_name="custom")
                    override_dict = payload
            except Exception as e:
                # print(f"⚠️  OVERRIDES: failed to load {OVERRIDES_CANON_PATH}: {e}")
                print(f"[OVERRIDES] Using command-line overrides via --overrides ({len(override_dict)} key(s))")


    # Resolve input videos
    if INPUT_MODE == "single":
        videos = [VIDEO_ROOT / SINGLE_CLIP_NAME]
    else:
        videos = list_mkvs(VIDEO_ROOT, config.MAX_FILES, config.RECURSIVE)

    if not videos:
        print(f" Found 0 .mkv files under VIDEO_ROOT: {VIDEO_ROOT}")
        sys.exit(0)

    print(f" Found {len(videos)} .mkv file{'s' if len(videos) != 1 else ''} under VIDEO_ROOT: {VIDEO_ROOT}")

    first_clip = videos[0]
    paths0 = sp.get_paths(str(first_clip), MODULE_NAME)

    log_dir0 = Path(paths0.get("LOG_DIR_P") or paths0["LOG_DIR"])

    if ENABLE_PREVIEW:
        # preview (choose source/preset)
        state.camera_snapshot = get_axis_ptz_snapshot()
        state.requested_preset = (state.camera_snapshot or {}).get("preset", "Home")

        _ = build_preview_with_options(
            first_clip_path=first_clip,
            log_dir=log_dir0,
            default_source=PREVIEW_DEFAULT_SOURCE,
            reference_dir=PREVIEW_REFERENCE_DIR,
            reference_preset=PREVIEW_REFERENCE_PRESET,
            axis_ip=None,
            axis_creds_path=None,
            auto_open=True,
            config=config,
            state=state,
        )
    else:
        print("[PREVIEW] disabled #1 by config toggle")


    if ENABLE_PREVIEW:
        # preview (choose source/preset)
        state.camera_snapshot = get_axis_ptz_snapshot()
        state.requested_preset = (state.camera_snapshot or {}).get("preset", "Home")

        _ = build_preview_with_options(
            first_clip_path=first_clip,
            log_dir=log_dir0,
            default_source=PREVIEW_DEFAULT_SOURCE,
            reference_dir=PREVIEW_REFERENCE_DIR,
            reference_preset=PREVIEW_REFERENCE_PRESET,
            axis_ip=None,
            axis_creds_path=None,
            auto_open=True,
            config=config,
            state=state,
        )
    else:
        print("[PREVIEW] disabled #2 by config toggle")

         

    _print_active_filter_settings(config, state)

    # Pre-run run_config.json (so aborted runs still have config)
    try:
        fps0, total0, w0, h0 = get_clip_info(first_clip)
        payload0 = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "video_name": first_clip.name,
            "video_path": str(first_clip),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "clip_info": {"fps": fps0, "total_frames": total0, "w": w0, "h": h0},
            "knobs": {k: getattr(config, k) for k in ALLOWED_OVERRIDES.keys() if hasattr(config, k)},
            "overrides": state.override_info,
            "camera_preset": state.requested_preset or "custom",
        }
        (log_dir0 / "run_config.json").write_text(json.dumps(payload0, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠ Could not write pre-run run_config.json: {e}")

    '''
    # proceed prompt
    try:
        # resp = input("Proceed with processing ALL clips? (y/N): ").strip().lower()
    # except (KeyboardInterrupt, EOFError):
        # resp = "n"
    # if resp not in ("y", "yes"):
        # print(" Aborted by user.")
        # sys.exit(0)
        
        if args.auto_yes:
            resp = "y"
        else:
            try:
                resp = input("Proceed with processing ALL clips? (y/N): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                resp = "n"

        if resp not in ("y", "yes"):
            print(" Aborted by user.")
            sys.exit(0)
    '''
    

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing {vp.name} …")
        try:
            p = paths0 if i == 1 else sp.get_paths(str(vp), MODULE_NAME)
            process_clip(vp, p, config, state)
        except Exception as e:
            print(f"    Error on {vp.name}: {e}")

if __name__ == "__main__":
    main()

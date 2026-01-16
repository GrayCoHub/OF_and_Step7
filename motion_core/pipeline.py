# pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import csv
import json
from datetime import datetime

import cv2
import numpy as np

from optical_flow_twoPass import STATUS_EVERY

'''

To run from REPL:

    from optical_flow_twoPass import Config
    from motion_core.pipeline import run_of_pipeline

    cfg = Config()
    result = run_of_pipeline(
        r"C:\AxisRecordings\Optical_Flow\videos\1-smallObject-R2L-bankingTurn_L2R.mkv",
        cfg,
    )

'''


# Core OF pipeline + packet/log writers + paths
from optical_flow_twoPass import (
    Config,
    State,
    MODULE_NAME,
    sp,
    build_keep_mask,
    compute_flow_and_mag,
    segment_from_kept,
    _segments_from_kept_frames,
    _write_frame_signal,
    _write_segment_summary,
    _write_run_manifest,
    _write_latest_pointer_files,
    _write_latest_run_dir_pointers,
    save_run_outputs,
    classify_target,
    auto_tuner,
    prescan,
)

from motion_core.filtering import (
    threshold_mag,
    apply_keep_mask,
    apply_morphology,
    build_keep_mask,
)


# near the top of pipeline.py, after imports
from pathlib import Path

# module_root = Path(__file__).resolve().parent
# module_root = sp.get_paths("dummy", MODULE_NAME)["MODULE_ROOT"]
module_root = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")





'''

# ============================================================
# PRESCAN + CLASSIFICATION + AUTO-TUNER (CLI-PARITY)
# ============================================================

# Explicit toggle (same as CLI)
if not getattr(config, "AUTO_TUNER_ENABLED", True):
    print("   [TUNER] disabled by config toggle")
    state.tuner_enabled = False
    state.tuner_reason = "disabled by config toggle"
    state.tuner_diff = None

else:
    # Determine whether any real overrides were used
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

                # Apply tuner and capture diff
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

    '''


# ======================================================================================
# PASS-1: DETECT (engine version, same logic as in optical_flow_twoPass)
# ======================================================================================

def detect_motion_frames(
    video_path: Path,
    config: Config,
    preset: str,
    state: State,
    logger: Optional[Any] = None,  # PerFrameLogger type, but we don't need it here
) -> Tuple[List[Tuple[int, int]], Set[int], int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"⚠️ Failed to open video: {video_path}")
        return [], set(), 0, 0.0, 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        return [], set(), total_frames, fps, frame_w, frame_h

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    kept_p1: Set[int] = set()


    for idx in range(1, total_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------------------
        # Optical flow
        # --------------------------------------------------------------
        flow, mag = compute_flow_and_mag(prev_gray, gray)
        motion_mask = mag

        # Optional debug tap
        dbg = (idx == 25)
        if dbg:
            print(f"[DBG] start: raw={np.count_nonzero(mag)}")

        # --------------------------------------------------------------
        # PASS‑1 gating pipeline
        # --------------------------------------------------------------

        # 1. threshold
        if config.GATING.get("USE_THRESHOLD", True):
            motion_mask = threshold_mag(motion_mask, float(config.MOTION_THRESHOLD))
            if dbg:
                print(f"[DBG] after threshold: {np.count_nonzero(motion_mask)}")

        # 2. keep mask
        if config.GATING.get("USE_KEEP_MASK", True):
            motion_mask = apply_keep_mask(motion_mask, keep_mask)
            if dbg:
                print(f"[DBG] after keep_mask: {np.count_nonzero(motion_mask)}")

        # 5. morphology
        if config.GATING.get("USE_MORPHOLOGY", True) and config.USE_MORPHOLOGY:
            motion_mask = apply_morphology(
                motion_mask,
                kernel_sz=config.MORPH_KERNEL_SZ,
                open_iters=config.MORPH_OPEN_ITERS,
                close_iters=config.MORPH_CLOSE_ITERS,
            )
            if dbg:
                print(f"[DBG] after morphology: {np.count_nonzero(motion_mask)}")

        # --------------------------------------------------------------
        # Pixel counts + kept decision
        # --------------------------------------------------------------
        active_raw = int(np.count_nonzero(mag))
        active_gated = int(np.count_nonzero(motion_mask))
        kept = active_gated >= int(config.MIN_MOTION_PIXELS)

        # Step‑3 splice: collect motion metrics
        if state.collect_motion_metrics:
            state.motion_pixels_per_frame.append(active_gated)

        # --------------------------------------------------------------
        # Flow statistics
        # --------------------------------------------------------------
        flow_dx_mean = 0.0
        flow_speed_mean = 0.0
        if active_gated > 0:
            m = (motion_mask > 0)
            flow_dx_mean = float(np.mean(flow[..., 0][m]))
            flow_speed_mean = float(np.mean(mag[m]))

        # --------------------------------------------------------------
        # Logging + bookkeeping
        # --------------------------------------------------------------
        if logger is not None:
            logger.log("p1", idx, active_raw, active_gated, kept, flow_dx_mean, flow_speed_mean)

        if kept:
            kept_p1.add(idx)

        prev_gray = gray

        # Status line
        if idx % STATUS_EVERY == 0:
            print(
                f"   > Frame {idx:05d}/{total_frames} | "
                f"raw={active_raw} gated={active_gated} kept={1 if kept else 0}"
            )

    cap.release()

    segments = segment_from_kept(sorted(kept_p1), config)
    return segments, kept_p1, total_frames, fps, frame_w, frame_h

            



# ======================================================================================
# PASS-2: RECOVER + REBUILD FINAL (engine version, same logic as in optical_flow_twoPass)
# ======================================================================================

def p2_recover_and_rebuild_segments(
    p1_segments: List[Tuple[int, int]],
    kept_p1: Set[int],
    video_path: Path,
    config: Config,
    preset: str,
    log_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int], Dict[str, Any], List[Dict[str, Any]]]:
    audit_rows: List[Dict[str, Any]] = []

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
        return [], set(), set(kept_p1), stats, audit_rows

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to open video for P2 recovery: {video_path}")
        stats = {
            "p1_segments": int(len(p1_segments)),
            "p2_windows": int(len(p1_segments)),
            "p1_frames": int(len(kept_p1)),
            "p2_recovered_frames": 0,
            "p2_recovered_in_pad": 0,
            "final_frames": int(len(kept_p1)),
            "final_segments": int(len(p1_segments)),
            "gain_frames": 0,
        }
        return p1_segments, set(), set(kept_p1), stats, audit_rows

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    keep_mask = build_keep_mask(config, (frame_h, frame_w), preset)

    pad = int(getattr(config, "P2_PAD_FRAMES", 0) or 0)
    thr2 = float(getattr(config, "MOTION_THRESHOLD_P2", config.MOTION_THRESHOLD))
    pix2 = int(getattr(config, "MIN_MOTION_PIXELS_P2", config.MIN_MOTION_PIXELS))

    kept_p2: Set[int] = set()
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

            raw_mask = (mag > thr2).astype(np.uint8)
            active_raw = int(cv2.countNonZero(raw_mask))

            if config.MASK_GATING_PASS2 == "hard":
                motion_mask = (raw_mask & keep_mask.astype(np.uint8))
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

            audit_rows.append({
                "frame_idx": int(idx),
                "in_p1_span": int(in_p1_span),
                "in_p2_pad": int(in_pad),
                "md_p1": int(idx in kept_p1),
                "md_p2": int(kept),
                "md_final": int((idx in kept_p1) or kept),
                "active_raw_p2": int(active_raw),
                "flow_energy_p2": float(flow_energy_p2),
                "flow_energy_norm_p2": float(flow_energy_norm_p2),
            })

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
                w.writerow([
                    "frame_idx", "in_p1_span", "in_p2_pad",
                    "md_p1", "md_p2", "md_final",
                    "active_raw_p2", "flow_energy_p2", "flow_energy_norm_p2"
                ])
                for r in audit_rows:
                    w.writerow([
                        r["frame_idx"],
                        r["in_p1_span"],
                        r["in_p2_pad"],
                        r["md_p1"],
                        r["md_p2"],
                        r["md_final"],
                        r["active_raw_p2"],
                        r["flow_energy_p2"],
                        r["flow_energy_norm_p2"],
                    ])
        except Exception as e:
            print(f"   ⚠️ failed to write P2 recovery audit CSV: {e}")

    print(
        f"   OK... P2 recovery: +{stats['gain_frames']} frame(s) "
        f"(P2-only={stats['p2_recovered_frames']}, in_pad={stats['p2_recovered_in_pad']}); "
        f"final_segments={stats['final_segments']}"
    )
    return final_segments, kept_p2, kept_final, stats, audit_rows


# ======================================================================================
# PASS-1 / PASS-2 wrappers + packet/log export
# ======================================================================================

def run_pass1(
    clip_path: str,
    config: Config,
    preset: str,
    state: State,
) -> Dict[str, Any]:
    video_path = Path(clip_path)

    segments, kept_p1, total_frames, fps, frame_w, frame_h = detect_motion_frames(
        video_path,
        config,
        preset,
        state,
        logger=None,
    )

    return {
        "segments": segments,
        "kept": kept_p1,
        "md_total": len(kept_p1),
        "total_frames": total_frames,
        "fps": fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def run_pass2(
    clip_path: str,
    config: Config,
    preset: str,
    p1_segments: List[Tuple[int, int]],
    kept_p1: Set[int],
    log_dir: Path,
) -> Dict[str, Any]:
    video_path = Path(clip_path)

    final_segments, kept_p2, kept_final, stats, audit_rows = p2_recover_and_rebuild_segments(
        p1_segments,
        kept_p1,
        video_path,
        config,
        preset,
        log_dir=log_dir,
    )

    return {
        "segments": final_segments,
        "kept_p2": kept_p2,
        "kept_final": kept_final,
        "md_total": len(kept_final),
        "stats": stats,
        "audit_rows": audit_rows,
    }


def _export_packets_and_pointers(
    clip_path: str,
    config: Config,
    state: State,
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    run_dir: Path,
    log_dir: Path,
) -> Dict[str, Any]:
    video_path = Path(clip_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save logs (segment logs + overrides + run_config)
    duration = 0.0  # if you later track processing time, pass real value
    save_run_outputs(
        log_dir,
        config,
        state,
        video_path,
        p1["segments"],
        duration,
        p1["fps"],
        p1["total_frames"],
        p1["frame_w"],
        p1["frame_h"],
    )

    # Build packet dirs under run_dir
    p1_dir = run_dir / "packet_of_p1"
    final_dir = run_dir / "packet_of_final"
    p1_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Frame signal + segment summary + manifest for P1
    rows_dict = p2["audit_rows"]  # contains md_p1, md_final, etc.
    md_frames_p1: Set[int] = set(
        r["frame_idx"] for r in rows_dict if int(r.get("md_p1", 0)) == 1
    ) if rows_dict else set()

    fs_p1 = _write_frame_signal(p1_dir, rows_dict, md_key="md_p1") if rows_dict else _write_frame_signal(p1_dir, [], md_key="md_p1")
    md_total_p1 = int(fs_p1["md_total"])

    seg_p1 = _write_segment_summary(p1_dir, p1["segments"], md_frames_p1)

    clip_info = {
        "fps": float(p1["fps"]),
        "total_frames": int(p1["total_frames"]),
        "w": int(p1["frame_w"]),
        "h": int(p1["frame_h"]),
    }

    knobs = {
        k: getattr(config, k)
        for k in getattr(config, "ALLOWED_OVERRIDES", {}).keys()
        if hasattr(config, k)
    } if hasattr(config, "ALLOWED_OVERRIDES") else {}

    overrides = getattr(state, "override_info", {}) or {}

    engine_flags = {
        "USE_THRESHOLD": config.GATING.get("USE_THRESHOLD", True),
        "USE_KEEP_MASK": config.GATING.get("USE_KEEP_MASK", True),
        "USE_MORPHOLOGY": config.USE_MORPHOLOGY,
        "ROI_FILTER_ENABLED": config.ROI_FILTER_ENABLED,
        "ROI_FILTER_MODE": getattr(config, "ROI_FILTER_MODE", None),
        "WATER_MASK_ENABLED": config.WATER_MASK_ENABLED,
        "WATER_MASK_MODE": getattr(config, "WATER_MASK_MODE", None),
    }

    _write_run_manifest(
        p1_dir,
        module=MODULE_NAME,
        variant="p1",
        video_path=str(video_path),
        timestamp=now,
        clip_info=clip_info,
        knobs=knobs,
        overrides=overrides,
        engine_flags=engine_flags,
        seg_summary=seg_p1,
        md_total=md_total_p1,        
    )

    # Frame signal + segment summary + manifest for FINAL
    md_frames_final: Set[int] = set(
        r["frame_idx"] for r in rows_dict if int(r.get("md_final", 0)) == 1
    ) if rows_dict else set()

    if rows_dict:
        fs_final = _write_frame_signal(final_dir, rows_dict, md_key="md_final")
        md_total_final = int(fs_final["md_total"])
    else:
        md_total_final = len(md_frames_final)

    seg_final = _write_segment_summary(final_dir, p2["segments"], md_frames_final)

    _write_run_manifest(
        final_dir,
        module=MODULE_NAME,
        variant="final",
        video_path=str(video_path),
        timestamp=now,
        clip_info=clip_info,
        knobs=knobs,
        overrides=overrides,
        seg_summary=seg_final,
        md_total=md_total_final,
    )

    # Pointer files for orchestrator
    # print(f"[DBG] pointer: writing _latest_run_dir → {run_dir}")
    # print(f"[DBG] pointer: writing _latest_packet_of_p1 → {p1_dir}")
    # print(f"[DBG] pointer: writing _latest_packet_of_final → {final_dir}")

    
    
    
    # print(f"[DBG] pointer: writing _latest_run_dir → {run_dir}")
    _write_latest_pointer_files(
        module_root,
        pointer_base="_latest_packet_of_p1",
        packet_dir=p1_dir,
        run_dir=run_dir,
        video_path=video_path,
    )
    # print(f"[DBG] pointer: writing _latest_run_dir → {run_dir}")
    _write_latest_pointer_files(
        module_root,
        pointer_base="_latest_packet_of_final",
        packet_dir=final_dir,
        run_dir=run_dir,
        video_path=video_path,
    )
    # print(f"[DBG] pointer: writing _latest_run_dir → {run_dir}")
    _write_latest_run_dir_pointers(
        module_root,
        run_dir=run_dir,
        video_path=video_path,
    )

    return {
        "p1_packet_dir": str(p1_dir),
        "final_packet_dir": str(final_dir),
        "run_dir": str(run_dir),
    }


# ======================================================================================
# Engine entry point (what the orchestrator will call)
# ======================================================================================

def run_of_pipeline(clip_path: str, config: Config) -> Dict[str, Any]:
    """
    Full OF pipeline engine:
      - PASS-1
      - PASS-2
      - logs
      - OF packets (p1 + final)
      - pointer JSONs

    Scoring and comparison are handled by packet_compare_scorecard.py + orchestrator_final.py.
    """
    video_path = Path(clip_path)

    # Derive preset the same way your CLI does; adjust if you have a different field.
    preset = getattr(config, "CAMERA_PRESET", "default")



def run_of_pipeline(clip_path: str, config: Config) -> Dict[str, Any]:
    video_path = Path(clip_path)
    preset = getattr(config, "CAMERA_PRESET", "default")

    # Build state
    state = State()

    # Resolve run paths
    paths = sp.get_paths(str(video_path), MODULE_NAME)
    run_dir: Path = paths["OUT_DIR_P"]
    log_dir: Path = paths["LOG_DIR_P"]

    print(f"\n[ENGINE] run_dir = {run_dir}")
    print(f"[ENGINE] log_dir = {log_dir}\n")



    # ============================================================
    # PRESCAN + CLASSIFICATION + AUTO-TUNER (CLI-PARITY)
    # ============================================================

    if not getattr(config, "AUTO_TUNER_ENABLED", True):
        print("   [TUNER] disabled by config toggle")
        state.tuner_enabled = False
        state.tuner_reason = "disabled by config toggle"
        state.tuner_diff = None

    else:
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
            try:
                cap_ps = cv2.VideoCapture(str(video_path))
                if cap_ps.isOpened():
                    metrics = prescan(cap_ps, config, preset, num_frames=40)
                    classification = classify_target(metrics)

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

    # PASS-1
    p1 = run_pass1(clip_path, config, preset, state)

    # PASS-2
    p2 = run_pass2(
        clip_path,
        config,
        preset,
        p1_segments=p1["segments"],
        kept_p1=p1["kept"],
        log_dir=log_dir,
    )

    # Logs + packets + pointers
    packet_info = _export_packets_and_pointers(
        clip_path,
        config,
        state,
        p1,
        p2,
        run_dir,
        log_dir,
    )

    return {
        "clip": clip_path,
        "p1": p1,
        "p2": p2,
        "packet_info": packet_info,
    }

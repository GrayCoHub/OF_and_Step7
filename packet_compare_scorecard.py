# packet_compare_scorecard.py

"""
To run from powershell:

python packet_compare_scorecard.py `
    --a LATEST:OF:p1 `
    --b LATEST:STEP7:baseline `
    --out "C:\Axis_code_projects\OF_vs_Step7\outputs\comparisons\big_bird_R2L_20260108_214426"


  
 """

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd

SCRIPT_VERSION = "2026-01-11"


from datetime import datetime
# ----------------------------
# Column detection heuristics
# ----------------------------
FRAME_CANDIDATES = ["frame", "frame_idx", "frame_index", "engine_frame", "idx"]



# SIGNAL_CANDIDATES = [
    # "signal", "frame_signal", "has_signal", "has_motion", "motion", "detected",
    # "is_detected", "active", "md", "pass", "p1", "p2"
# ]


SIGNAL_CANDIDATES = [
    "md_flag",              # Step7 observed hits (preferred)
    "signal", "frame_signal",
    "has_signal", "has_motion",
    "motion", "detected", "is_detected",
    "active_pixels",        # if you ever store “pixels of motion”
    "active", "md", "pass", "p1", "p2",
    "in_primary_span"       # Step7 span/retro membership (fallback)
]





SEG_START_CAND = ["start_frame", "start", "seg_start", "begin_frame", "frame_start"]
SEG_END_CAND   = ["end_frame", "end", "seg_end", "finish_frame", "frame_end"]
SEG_LEN_CAND   = ["segment_length", "length", "len", "frames", "n_frames"]


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _pick_frame_and_signal(df: pd.DataFrame) -> Tuple[str, str]:
    # frame col
    fcol = _pick_col(list(df.columns), FRAME_CANDIDATES)
    if fcol is None:
        # fallback: first integer-like column
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
                fcol = c
                break
    if fcol is None:
        raise ValueError("Could not identify frame column in frame_signal.csv")

    # signal col
    scol = _pick_col(list(df.columns), SIGNAL_CANDIDATES)
    if scol is None:
        # fallback: the first non-frame numeric/bool column
        for c in df.columns:
            if c == fcol:
                continue
            if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
                scol = c
                break
    if scol is None:
        raise ValueError("Could not identify signal column in frame_signal.csv")

    return fcol, scol


# def load_packet(packet_dir: Path) -> Dict:

def load_packet(packet_dir: Path, signal_col_override: str | None = None) -> Dict:
    """
    Load a packet directory containing:
      - frame_signal.csv
      - segment_summary.csv
      - run_manifest.json

    Normalize:
      - frames → DataFrame with {frame:int, detected:int}
      - segments → DataFrame with {start_frame, end_frame, segment_length}
    """

    packet_dir = packet_dir.resolve()
    fs_path = packet_dir / "frame_signal.csv"
    seg_path = packet_dir / "segment_summary.csv"
    man_path = packet_dir / "run_manifest.json"

    # --- existence checks ---
    if not fs_path.exists():
        raise FileNotFoundError(f"Missing {fs_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing {seg_path}")
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}")

    # --- load frame_signal ---
    df_fs = pd.read_csv(fs_path)

    # frame column
    fcol = _pick_col(list(df_fs.columns), FRAME_CANDIDATES)
    if fcol is None:
        raise ValueError(
            f"Could not identify frame column in {fs_path}. "
            f"Columns={list(df_fs.columns)}"
        )

    # signal column
    if signal_col_override:
        if signal_col_override not in df_fs.columns:
            raise ValueError(
                f"Requested signal col '{signal_col_override}' not found in {fs_path}. "
                f"Columns={list(df_fs.columns)}"
            )
        scol = signal_col_override
    else:
        _f2, scol = _pick_frame_and_signal(df_fs)

    # normalize frame_signal
    df = df_fs[[fcol, scol]].copy()
    df.rename(columns={fcol: "frame", scol: "detected_raw"}, inplace=True)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")

    if pd.api.types.is_bool_dtype(df["detected_raw"]):
        df["detected"] = df["detected_raw"].astype(int)
    else:
        df["detected"] = (
            pd.to_numeric(df["detected_raw"], errors="coerce").fillna(0) > 0
        ).astype(int)

    df = (
        df.dropna(subset=["frame"])
          .astype({"frame": int})
          .sort_values("frame")
          .reset_index(drop=True)
    )

    # --- load segment_summary ---
    df_seg = pd.read_csv(seg_path)
    s_start = _pick_col(list(df_seg.columns), SEG_START_CAND)
    s_end   = _pick_col(list(df_seg.columns), SEG_END_CAND)
    s_len   = _pick_col(list(df_seg.columns), SEG_LEN_CAND)

    if s_start and s_end:
        seg_norm = df_seg[[s_start, s_end] + ([s_len] if s_len else [])].copy()
        seg_norm.rename(columns={s_start: "start_frame", s_end: "end_frame"}, inplace=True)
        if s_len:
            seg_norm.rename(columns={s_len: "segment_length"}, inplace=True)
        else:
            seg_norm["segment_length"] = (
                seg_norm["end_frame"] - seg_norm["start_frame"] + 1
            )
        seg_norm["start_frame"] = pd.to_numeric(seg_norm["start_frame"], errors="coerce")
        seg_norm["end_frame"] = pd.to_numeric(seg_norm["end_frame"], errors="coerce")
        seg_norm["segment_length"] = pd.to_numeric(seg_norm["segment_length"], errors="coerce")
        seg_norm = (
            seg_norm.dropna(subset=["start_frame", "end_frame"])
                    .astype(int)
                    .sort_values("start_frame")
                    .reset_index(drop=True)
        )
    else:
        seg_norm = compute_segments_from_frames(df)

    # --- manifest ---
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    # --- schema audit ---
    schema_audit = {
        "frame_signal_columns": list(df_fs.columns),
        "picked_frame_col": fcol,
        "picked_signal_col": scol,
        "segment_columns": list(df_seg.columns),
        "picked_seg_start": s_start,
        "picked_seg_end": s_end,
        "picked_seg_len": s_len,
    }

    # --- final return structure ---
    return {
        "dir": str(packet_dir),
        "paths": {
            "dir": str(packet_dir),
            "manifest_path": str(man_path),
            "frame_signal_path": str(fs_path),
            "segment_summary_path": str(seg_path),
        },
        "meta": {
            "module": manifest.get("module"),
            "variant": manifest.get("variant"),
            "video_name": manifest.get("video_name"),
            "video_path": manifest.get("video_path"),
            "timestamp": manifest.get("timestamp"),
            "clip_info": manifest.get("clip_info"),
            "primary_span": manifest.get("primary_span"),
            "span": manifest.get("span"),
            "md_total": manifest.get("md_total"),
            "primary_density": manifest.get("primary_density"),
        },
        "manifest": manifest,
        "frames": df,
        "segments": seg_norm,
        "schema_audit": schema_audit,
    }

    
    
def _print_packet_summary(label: str, packet: Dict[str, Any]) -> None:
    """
    Print a concise, deterministic summary of what's actually being compared.
    Must not raise due to pandas "truth value" ambiguity.
    """
    meta = packet.get("meta", {}) or {}
    paths = packet.get("paths", {}) or {}

    print("\n" + "=" * 86)
    # print(f"📦 PACKET {label}")
    print(f"[PACKET {label}]")

    print("-" * 86)
    print(f"dir            : {paths.get('dir')}")
    print(f"manifest_path  : {paths.get('manifest_path')}")
    print(f"frame_signal   : {paths.get('frame_signal_path')}")
    print(f"segment_summary: {paths.get('segment_summary_path')}")
    print("-" * 86)

    module = meta.get("module")
    variant = meta.get("variant")
    video_name = meta.get("video_name")
    ts = meta.get("timestamp")
    clip = meta.get("clip_info") or {}
    fps = clip.get("fps")
    total_frames = clip.get("total_frames")
    w = clip.get("w")
    h = clip.get("h")

    primary_span = meta.get("primary_span")
    # normalize primary_span to [start,end] if possible
    ps_norm = None
    if isinstance(primary_span, (list, tuple)) and len(primary_span) == 2:
        ps_norm = [int(primary_span[0]), int(primary_span[1])]
    elif isinstance(primary_span, dict):
        if "start" in primary_span and "end" in primary_span:
            ps_norm = [int(primary_span["start"]), int(primary_span["end"])]
    elif isinstance(primary_span, str):
        # tolerate "start,end" or "[start,end]"
        m = re.findall(r"-?\d+", primary_span)
        if len(m) >= 2:
            ps_norm = [int(m[0]), int(m[1])]

    span = meta.get("span")
    md_total = meta.get("md_total")
    dens = meta.get("primary_density")

    print(f"module/variant : {module} / {variant}")
    print(f"video          : {video_name}")
    print(f"timestamp      : {ts}")
    print(f"clip_info      : fps={fps} total_frames={total_frames} size={w}x{h}")
    print(f"primary_span   : {ps_norm if ps_norm is not None else primary_span}  span={span}  md_total={md_total}  density={dens}")
    print("-" * 86)

    # frame signal
    fdf = packet.get("frame_df")
    if fdf is None:
        fdf = packet.get("frames")
    try:
        if fdf is not None and getattr(fdf, "shape", None) is not None:
            n_rows = int(fdf.shape[0])
            col_frame = "frame" if "frame" in fdf.columns else ("frame_idx" if "frame_idx" in fdf.columns else None)
            col_det = "detected" if "detected" in fdf.columns else ("md_final" if "md_final" in fdf.columns else None)
            md_sum = int(fdf[col_det].sum()) if (col_det is not None and n_rows > 0) else None
            if col_frame is not None and n_rows > 0:
                fmin = int(fdf[col_frame].min())
                fmax = int(fdf[col_frame].max())
            else:
                fmin, fmax = None, None
            print(f"frame_signal rows: {n_rows} | md_sum={md_sum} | frame_range=({fmin}..{fmax})")
        else:
            print("frame_signal rows: (missing)")
    except Exception as e:
        print(f"⚠️ frame_signal summary failed: {e}")

    # segment summary
    sdf = packet.get("seg_df")
    if sdf is None:
        sdf = packet.get("segments")
    try:
        if sdf is not None and getattr(sdf, "shape", None) is not None:
            n_seg = int(sdf.shape[0])
            print(f"segment_summary rows: {n_seg}")
            if n_seg > 0:
                r0 = sdf.iloc[0].to_dict()
                # keep just the most useful bits
                keep = {k: r0.get(k) for k in ["segment_id", "start_frame", "end_frame", "span", "md", "density", "gap_frames", "longest_gap"] if k in r0}
                if not keep:
                    keep = r0
                print(f"first segment   : {keep}")
        else:
            print("segment_summary rows: (missing)")
    except Exception as e:
        print(f"⚠️ segment_summary summary failed: {e}")

def compute_segments_from_frames(df_frames: pd.DataFrame) -> pd.DataFrame:
    # assumes df_frames has "frame" and "detected" 0/1
    det = df_frames[df_frames["detected"] == 1]["frame"].to_list()
    if not det:
        return pd.DataFrame(columns=["start_frame", "end_frame", "segment_length"])

    starts = [det[0]]
    ends = []
    for i in range(1, len(det)):
        if det[i] != det[i - 1] + 1:
            ends.append(det[i - 1])
            starts.append(det[i])
    ends.append(det[-1])

    seg = pd.DataFrame({"start_frame": starts, "end_frame": ends})
    seg["segment_length"] = seg["end_frame"] - seg["start_frame"] + 1
    return seg


def windows_from_mask(frames: pd.DataFrame, mask_col: str) -> pd.DataFrame:
    # frames: has 'frame' and mask_col 0/1
    df = frames.sort_values("frame").copy()
    on = df[df[mask_col] == 1]["frame"].to_list()
    if not on:
        return pd.DataFrame(columns=["start_frame", "end_frame", "length"])

    starts = [on[0]]
    ends = []
    for i in range(1, len(on)):
        if on[i] != on[i - 1] + 1:
            ends.append(on[i - 1])
            starts.append(on[i])
    ends.append(on[-1])

    out = pd.DataFrame({"start_frame": starts, "end_frame": ends})
    out["length"] = out["end_frame"] - out["start_frame"] + 1
    return out


def segment_stats(seg: pd.DataFrame) -> Dict[str, float]:
    if seg is None or seg.empty:
        return {
            "segment_count": 0,
            "total_segment_frames": 0,
            "mean_segment_len": 0.0,
            "median_segment_len": 0.0,
            "min_segment_len": 0,
            "max_segment_len": 0,
            "first_seg_start": None,
            "last_seg_end": None,
        }

    # --- merge overlapping spans ---
    spans = sorted(
        (int(r.start_frame), int(r.end_frame))
        for _, r in seg.iterrows()
    )

    merged = []
    cur_start, cur_end = spans[0]

    for start, end in spans[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))

    # unique coverage
    unique_total = sum((end - start + 1) for start, end in merged)

    # original stats still computed from raw segments
    lens = seg["segment_length"]

    return {
        "segment_count": int(len(seg)),
        "total_segment_frames": int(unique_total),   # <-- FIXED
        "mean_segment_len": float(lens.mean()),
        "median_segment_len": float(lens.median()),
        "min_segment_len": int(lens.min()),
        "max_segment_len": int(lens.max()),
        "first_seg_start": int(seg["start_frame"].iloc[0]),
        "last_seg_end": int(seg["end_frame"].iloc[-1]),
    }


def compare_packets(packet_a: Dict, packet_b: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    A = packet_a["frames"].rename(columns={"detected": "A"})
    B = packet_b["frames"].rename(columns={"detected": "B"})

    # Outer join so we don't lose frames if one side is missing rows
    merged = pd.merge(A, B, on="frame", how="outer").fillna(0).astype({"A": int, "B": int})
    merged["both"] = ((merged["A"] == 1) & (merged["B"] == 1)).astype(int)
    merged["either"] = ((merged["A"] == 1) | (merged["B"] == 1)).astype(int)
    merged["A_only"] = ((merged["A"] == 1) & (merged["B"] == 0)).astype(int)
    merged["B_only"] = ((merged["A"] == 0) & (merged["B"] == 1)).astype(int)

    a_count = int(merged["A"].sum())
    b_count = int(merged["B"].sum())
    both_count = int(merged["both"].sum())
    either_count = int(merged["either"].sum())
    jaccard = (both_count / either_count) if either_count else 1.0

    # Treat B as "reference" for directional precision/recall (and vice versa)
    # (not ground truth—just a symmetric way to quantify disagreement)
    precision_A_vs_B = (both_count / a_count) if a_count else 1.0
    recall_A_vs_B    = (both_count / b_count) if b_count else 1.0

    # disagreement windows
    win_a_only = windows_from_mask(merged[["frame", "A_only"]], "A_only")
    win_a_only["type"] = "A_only"
    win_b_only = windows_from_mask(merged[["frame", "B_only"]], "B_only")
    win_b_only["type"] = "B_only"
    windows = pd.concat([win_a_only, win_b_only], ignore_index=True).sort_values(["length", "start_frame"], ascending=[False, True])

    # segment stats from provided segment_summary (or computed fallback)
    segA = segment_stats(packet_a["segments"])
    segB = segment_stats(packet_b["segments"])

    score = {
        "packet_A_dir": packet_a["dir"],
        "packet_B_dir": packet_b["dir"],
        "A_detected_frames": a_count,
        "B_detected_frames": b_count,
        "both_detected_frames": both_count,
        "either_detected_frames": either_count,
        "jaccard_overlap": jaccard,
        "precision_A_vs_B": precision_A_vs_B,
        "recall_A_vs_B": recall_A_vs_B,
        **{f"A_{k}": v for k, v in segA.items()},
        **{f"B_{k}": v for k, v in segB.items()},
    }

    audit = {
        "A": packet_a["schema_audit"],
        "B": packet_b["schema_audit"],
        "manifest_A": packet_a["manifest"],
        "manifest_B": packet_b["manifest"],
    }

    return pd.DataFrame([score]), windows.reset_index(drop=True), audit



# --------------------------------------------------------------------------------------
# LATEST token + pointer resolution
# --------------------------------------------------------------------------------------

def _default_outputs_root() -> Path:
    win = Path(r"C:\\Axis_code_projects\\OF_vs_Step7\\outputs")
    return win if win.exists() else (Path.cwd() / "outputs")


def _scan_latest_packet_dir(module_root: Path, packet_dirname: str) -> Optional[Path]:
    """Fallback: find the most recently modified packet directory under module_root."""
    if not module_root.exists():
        return None

    best: Optional[Tuple[float, Path]] = None
    # rglob is fine here; folder counts are small.
    for p in module_root.rglob(packet_dirname):
        if not p.is_dir():
            continue
        try:
            ts = p.stat().st_mtime
        except Exception:
            continue
        if best is None or ts > best[0]:
            best = (ts, p)

    return best[1] if best else None


def _read_pointer_path(ptr: Path) -> Optional[Path]:
    """Read pointer file that may be JSON or plain-text."""
    try:
        if ptr.suffix.lower() == ".json":
            data = json.loads(ptr.read_text(encoding="utf-8"))
            # Accept a few key names for backward/forward compat.
            for k in ("packet_dir", "packet_dir_abs", "path", "packet_path", "dir"):
                if k in data and data[k]:
                    return Path(str(data[k]))
            # Last resort: if the json itself is a string.
            if isinstance(data, str) and data.strip():
                return Path(data.strip())
            return None
        # .txt (or anything else): first non-empty line is the path.
        text = ptr.read_text(encoding="utf-8", errors="ignore").strip()
        return Path(text) if text else None
    except Exception:
        return None




OF_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")
STEP7_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\Step7_full")


def resolve_packet_path(tag: str) -> Path:
    """
    Resolve either:
      - a literal packet directory path
      - a pointer tag like:
            LATEST:OF:p1
            LATEST:OF:final
            LATEST:STEP7:baseline
            LATEST:STEP7:retro
    Returns a Path to the packet directory.
    """

    # ------------------------------------------------------------
    # 1. Literal path (user passed an absolute directory)
    # ------------------------------------------------------------
    if not tag.startswith("LATEST:"):
        p = Path(tag).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Packet path not found: {p}")
        return p

    # ------------------------------------------------------------
    # 2. Pointer tag: LATEST:<family>:<variant>
    # ------------------------------------------------------------
    try:
        _, family, variant = tag.split(":")
    except ValueError:
        raise ValueError(f"Invalid pointer tag format: {tag}")

    family = family.upper()
    variant = variant.lower()

    # ------------------------------------------------------------
    # 3. Map pointer tag → pointer JSON file
    # ------------------------------------------------------------
    if family == "OF":
        pointer_file = OF_ROOT / f"_latest_packet_of_{variant}.json"
    elif family == "STEP7":
        pointer_file = STEP7_ROOT / f"_latest_packet_step7_{variant}.json"
    else:
        raise ValueError(f"Unknown packet family in tag: {family}")

    if not pointer_file.exists():
        raise FileNotFoundError(f"Pointer file not found: {pointer_file}")

    # ------------------------------------------------------------
    # 4. Load pointer JSON → extract packet_dir
    # ------------------------------------------------------------
    data = json.loads(pointer_file.read_text(encoding="utf-8"))

    packet_dir = Path(data["packet_dir"]).resolve()

    if not packet_dir.exists():
        raise FileNotFoundError(
            f"Packet directory from pointer does not exist: {packet_dir}"
        )

    return packet_dir

    
    
    
def resolve_packet_arg(arg: str, root: Path) -> Path:
    raw = (arg or "").strip()
    if raw.upper().startswith("POINTER:"):
        p = Path(raw.split(":", 1)[1]).expanduser()
        if not p.exists():
            raise SystemExit(f"Pointer file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        pkt = Path(data.get("packet_dir") or "")
        if not pkt.exists():
            raise SystemExit(f"Pointer packet_dir does not exist: {pkt} (from {p})")
        return pkt

    if raw.upper().startswith("LATEST:"):
        parts = raw.split(":")
        # LATEST:<algo>:<variant>
        if len(parts) < 3:
            raise SystemExit(f"Bad LATEST token: {raw!r} (use LATEST:OF:p1 or LATEST:STEP7:baseline)")
        _, algo, variant = parts[0], parts[1], parts[2]
        return _resolve_latest_pointer(root, algo, variant)

    p = Path(raw).expanduser()
    if not p.exists():
        raise SystemExit(f"Packet path not found: {p}")
    return p


def _auto_out_dir(root: Path, packet_a: Dict[str, Any], packet_b: Dict[str, Any]) -> Path:
    vid = (packet_a.get("video_name") or packet_b.get("video_name") or "comparison")
    stem = Path(str(vid)).stem
    a_var = (packet_a.get("variant") or "A")
    b_var = (packet_b.get("variant") or "B")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "comparisons" / f"{stem}_{a_var}_vs_{b_var}_{stamp}"




# --------------------------------------------------------------------------------------

def compare_and_write_scorecard(pkt_a: Dict[str, Any], pkt_b: Dict[str, Any], out_dir: Path, title: Optional[str] = None) -> None:
    """Run comparison and write artifacts to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_df, windows_df, audit = compare_packets(pkt_a, pkt_b)

    # Write outputs
    score_path = out_dir / "scorecard.csv"
    windows_path = out_dir / "windows.csv"
    audit_path = out_dir / "audit.json"

    try:
        score_df.to_csv(score_path, index=False)
    except Exception as e:
        print(f"⚠️ Failed to write {score_path}: {e}")
    try:
        windows_df.to_csv(windows_path, index=False)
    except Exception as e:
        print(f"⚠️ Failed to write {windows_path}: {e}")
    try:
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Failed to write {audit_path}: {e}")

    # Console summary
    try:
        top = score_df.iloc[0].to_dict() if len(score_df) else {}
        print("\n=== SCORE SUMMARY ===")
        if title:
            print(f"Title: {title}")
        for k in [
            "jaccard_md", "f1_b_as_ref", "precision_b_as_ref", "recall_b_as_ref",
            "md_a", "md_b", "md_intersection", "md_union",
            "primary_overlap_iou", "span_a", "span_b",
        ]:
            if k in top:
                print(f"{k:20s}: {top[k]}")
        print(f"\nWrote: {score_path}")
        print(f"Wrote: {windows_path}")
        print(f"Wrote: {audit_path}")
    except Exception as e:
        print(f"⚠️ Failed to print score summary: {e}")



from pathlib import Path
import json

OF_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")
STEP7_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\Step7_full")


def _resolve_latest_pointer(root: Path, algo: str, variant: str) -> Path:
    """
    Resolve LATEST:<ALGO>:<VARIANT> into a concrete packet directory.
    Uses only pointer JSON files written by OF and Step7.
    """

    algo_u = algo.strip().upper()
    variant_l = variant.strip().lower()

    if algo_u == "OF":
        pointer_file = OF_ROOT / f"_latest_packet_of_{variant_l}.json"

    elif algo_u == "STEP7":
        pointer_file = STEP7_ROOT / f"_latest_packet_step7_{variant_l}.json"

    else:
        raise ValueError(f"Unsupported LATEST algo: {algo!r} (use OF or STEP7)")

    if not pointer_file.exists():
        raise FileNotFoundError(f"Pointer file not found: {pointer_file}")

    data = json.loads(pointer_file.read_text(encoding="utf-8"))
    packet_dir = Path(data["packet_dir"]).resolve()

    if not packet_dir.exists():
        raise FileNotFoundError(
            f"Packet directory from pointer does not exist: {packet_dir}"
        )

    return packet_dir





def main():
    print(f"[packet_compare_scorecard] v{SCRIPT_VERSION} | file={__file__}")
    ap = argparse.ArgumentParser(description="Compare two packet folders and emit a scorecard.")

    ap.add_argument(
        "--root",
        type=str,
        default=str(_default_outputs_root()),
        help=(
            "Outputs root used to resolve LATEST:* tokens and to place AUTO comparison outputs. "
            "Default: C:\\Axis_code_projects\\OF_vs_Step7\\outputs (if it exists), else ./outputs"
        ),
    )

    ap.add_argument(
        "--a",
        type=str,
        required=True,
        help=(
            "Packet A directory or token. Examples: "
            "C:/.../packet_of_p1 OR LATEST:OF:p1 OR POINTER:C:/.../_latest_packet_of_p1.json"
        ),
    )

    ap.add_argument(
        "--b",
        type=str,
        required=True,
        help=(
            "Packet B directory or token. Examples: "
            "C:/.../packet_step7_baseline OR LATEST:STEP7:baseline"
        ),
    )

    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help=(
            "Output directory for the comparison report. Use AUTO to create a dated folder under <root>/comparisons."
        ),
    )

    ap.add_argument("--title", type=str, default="", help="Optional title to embed in the report.")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    # Resolve packet dirs (support LATEST: and POINTER: tokens)
    a_dir = resolve_packet_arg(args.a, root)
    b_dir = resolve_packet_arg(args.b, root)

    print("\n=== RESOLVED INPUTS ===")
    print(f"A: {a_dir}")
    print(f"B: {b_dir}")

    pkt_a = load_packet(a_dir)
    if pkt_a.get('load_error'):
        print(f"\n❌ Failed to load PACKET A: {pkt_a.get('load_error')}\n")
        sys.exit(2)
    pkt_b = load_packet(b_dir)
    if pkt_b.get('load_error'):
        print(f"\n❌ Failed to load PACKET B: {pkt_b.get('load_error')}\n")
        sys.exit(2)

    # --- explicit packet summaries (so it's obvious what got compared) ---
    _print_packet_summary('A', pkt_a)
    _print_packet_summary('B', pkt_b)


    # Decide output dir (AUTO happens *after* load so we can name it from video/variants)
    if str(args.out).strip().upper() == "AUTO":
        out_dir = _auto_out_dir(root, pkt_a, pkt_b)
    else:
        out_dir = Path(args.out).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_dir}\n")

    # Compare + write report
    compare_and_write_scorecard(pkt_a, pkt_b, out_dir, title=args.title)


if __name__ == "__main__":
    main()

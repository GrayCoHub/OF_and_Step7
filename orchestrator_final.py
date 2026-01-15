# ORCHESTRATOR — FINAL


#!/usr/bin/env python3
"""
ORCHESTRATOR — FINAL, FILENAME‑AGNOSTIC VERSION
------------------------------------------------

To run:   python .\orchestrator_final.py


This orchestrator performs a complete comparison run using ONLY pointer tags:

    PRIMARY MODE (default):
        OF P1          vs   Step7 Baseline
        --a LATEST:OF:p1
        --b LATEST:STEP7:baseline

    ADVANCED MODE (optional):
        OF FINAL       vs   Step7 RETRO
        --a LATEST:OF:final
        --b LATEST:STEP7:retro

All packet paths are resolved through pointer JSONs written by OF and Step7.
No run‑dir inference. No filename dependence. No scanning.

Outputs:
    <outputs>/comparisons/<timestamped_run>/scorecard.csv
    <outputs>/comparisons/<timestamped_run>/windows.csv
    <outputs>/comparisons/<timestamped_run>/audit.json
"""

import subprocess
from pathlib import Path
from datetime import datetime
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# FIXED ROOTS — these never change
# ---------------------------------------------------------------------------
OF_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\OF_HO")
STEP7_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\Step7_full")
COMPARE_ROOT = Path(r"C:\Axis_code_projects\OF_vs_Step7\outputs\comparisons")


# ---------------------------------------------------------------------------
# POINTER RESOLVER — canonical, filename‑agnostic
# ---------------------------------------------------------------------------
def resolve_latest_pointer(algo: str, variant: str) -> Path:
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
        raise ValueError(f"Unsupported algo: {algo}")

    if not pointer_file.exists():
        raise FileNotFoundError(f"Pointer file missing: {pointer_file}")

    data = json.loads(pointer_file.read_text(encoding="utf-8"))
    packet_dir = Path(data["packet_dir"]).resolve()

    if not packet_dir.exists():
        raise FileNotFoundError(f"Packet directory missing: {packet_dir}")

    return packet_dir


# ---------------------------------------------------------------------------
# RUN SCORECARD
# ---------------------------------------------------------------------------
def run_scorecard(packet_a: Path, packet_b: Path, out_dir: Path) -> None:
    """
    Invoke packet_compare_scorecard.py with resolved packet paths.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "packet_compare_scorecard.py",
        "--a", str(packet_a),
        "--b", str(packet_b),
        "--out", str(out_dir),
    ]

    print("\n=== Running Scorecard ===")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)



def summarize_packet(label: str, packet_dir: Path) -> None:
    """
    Print a compact sanity-check summary of a packet before comparison.
    """
    man_path = packet_dir / "run_manifest.json"
    seg_path = packet_dir / "segment_summary.csv"
    fs_path  = packet_dir / "frame_signal.csv"

    # Load manifest
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{label}] ERROR: Could not read manifest: {e}")
        return

    # Load segment summary
    try:
        import pandas as pd
        df_seg = pd.read_csv(seg_path)
        seg_count = len(df_seg)
        if seg_count > 0:
            first_seg = df_seg.iloc[0].to_dict()
        else:
            first_seg = None
    except Exception:
        seg_count = None
        first_seg = None

    # Load frame signal
    try:
        df_fs = pd.read_csv(fs_path)
        md_total = df_fs.iloc[:, 1].sum()  # second column = detected
        frame_min = df_fs.iloc[:, 0].min()
        frame_max = df_fs.iloc[:, 0].max()
    except Exception:
        md_total = None
        frame_min = None
        frame_max = None

    print(f"\n[{label}] Packet Summary")
    print(f"  dir          : {packet_dir}")
    print(f"  video        : {manifest.get('video_name')}")
    print(f"  timestamp    : {manifest.get('timestamp')}")
    print(f"  variant      : {manifest.get('variant')}")
    print(f"  md_total     : {md_total}")
    print(f"  segments     : {seg_count}")
    print(f"  first_segment: {first_seg}")
    print(f"  frame_range  : ({frame_min}..{frame_max})")


def check_packet_integrity(label: str, packet_dir: Path) -> bool:
    """
    Validate that a packet directory contains the required files and
    that the contents are structurally sane. Returns True if OK, False otherwise.
    """
    print(f"\n[{label}] Integrity Check")
    ok = True

    man_path = packet_dir / "run_manifest.json"
    fs_path  = packet_dir / "frame_signal.csv"
    seg_path = packet_dir / "segment_summary.csv"

    # --- Required files ---
    for p in [man_path, fs_path, seg_path]:
        if not p.exists():
            print(f"  ERROR: Missing required file: {p}")
            ok = False

    if not ok:
        return False

    # --- Load manifest ---
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERROR: Could not parse manifest: {e}")
        return False

    # Required manifest fields
    required_fields = ["video_name", "timestamp", "variant", "md_total", "clip_info"]
    for f in required_fields:
        if f not in manifest:
            print(f"  ERROR: Manifest missing field: {f}")
            ok = False

    # --- Load frame_signal ---
    try:
        import pandas as pd
        df_fs = pd.read_csv(fs_path)
    except Exception as e:
        print(f"  ERROR: Could not read frame_signal.csv: {e}")
        return False

    # Expect at least 2 columns: frame + detected
    if df_fs.shape[1] < 2:
        print("  ERROR: frame_signal.csv has fewer than 2 columns")
        ok = False

    # Frame sanity
    try:
        frames = df_fs.iloc[:, 0].astype(int)
        if frames.min() < 0 or frames.max() > 100000:
            print("  WARNING: Frame range looks suspicious")
        if not frames.is_monotonic_increasing:
            print("  ERROR: Frame numbers are not monotonic increasing")
            ok = False
    except Exception:
        print("  ERROR: Could not parse frame numbers")
        ok = False

    # MD sanity
    try:
        detected = df_fs.iloc[:, 1].astype(int)
        md_total = detected.sum()
        if md_total <= 0:
            print("  WARNING: md_total is zero or negative")
    except Exception:
        print("  ERROR: Could not parse detected column")
        ok = False

    # --- Load segment_summary ---
    try:
        df_seg = pd.read_csv(seg_path)
    except Exception as e:
        print(f"  ERROR: Could not read segment_summary.csv: {e}")
        return False

    if len(df_seg) == 0:
        print("  WARNING: segment_summary.csv has zero rows")

    # Segment sanity
    if "start_frame" in df_seg.columns and "end_frame" in df_seg.columns:
        bad = df_seg[df_seg["start_frame"] > df_seg["end_frame"]]
        if len(bad) > 0:
            print("  ERROR: Found segments where start_frame > end_frame")
            ok = False

    print("  Integrity: OK" if ok else "  Integrity: FAILED")
    return ok


def plot_md_comparison(packet_a: Path, packet_b: Path, out_dir: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import json

    # ------------------------------------------------------------
    # LOAD MANIFESTS — this is the exact insertion point
    # ------------------------------------------------------------
    man_a = json.loads((packet_a / "run_manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((packet_b / "run_manifest.json").read_text(encoding="utf-8"))

    variant_a = man_a.get("variant", "unknown")
    variant_b = man_b.get("variant", "unknown")
    video_name = man_a.get("video_name", "unknown")

    # ------------------------------------------------------------
    # LOAD FRAME SIGNALS (existing code)
    # ------------------------------------------------------------
    fs_a = pd.read_csv(packet_a / "frame_signal.csv")
    fs_b = pd.read_csv(packet_b / "frame_signal.csv")

    frames_a = fs_a.iloc[:, 0].astype(int)
    md_a     = fs_a.iloc[:, 1].astype(int)

    frames_b = fs_b.iloc[:, 0].astype(int)
    md_b     = fs_b.iloc[:, 1].astype(int)

    # Align on a common frame axis
    f_min = min(frames_a.min(), frames_b.min())
    f_max = max(frames_a.max(), frames_b.max())
    frame_axis = range(f_min, f_max + 1)

    # Reindex MD signals to common axis
    md_a_aligned = pd.Series(0, index=frame_axis)
    md_b_aligned = pd.Series(0, index=frame_axis)

    md_a_aligned.loc[frames_a] = md_a.values
    md_b_aligned.loc[frames_b] = md_b.values

    overlap = (md_a_aligned & md_b_aligned)

    plt.figure(figsize=(14, 4))
    plt.plot(frame_axis, md_a_aligned, label="OF MD", color="blue", linewidth=1.5)
    plt.plot(frame_axis, md_b_aligned, label="Step7 MD", color="orange", linewidth=1.5)
    plt.fill_between(frame_axis, 0, overlap, color="green", alpha=0.3, label="Overlap")

    # plt.title("MD Comparison: OF vs Step7")
    plt.title(
        f"{video_name} — MD Comparison: OF {variant_a.upper()} vs Step7 {variant_b.upper()}"
    )

    plt.xlabel("Frame")
    plt.ylabel("MD (0/1)")
    plt.legend()
    plt.tight_layout()

        
    out_path = out_dir / "md_comparison.png"
    plt.savefig(out_path)
    plt.close()

    print(f"  Wrote MD visualization: {out_path}")

    # Auto-open the image using the OS default viewer
    try:
        import os, platform, subprocess
        if platform.system() == "Windows":
            os.startfile(out_path)  # Windows-native
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", out_path])
        else:  # Linux / other
            subprocess.Popen(["xdg-open", out_path])
        print("  Opened MD visualization.")
    except Exception as e:
        print(f"  WARNING: Could not auto-open image: {e}")

    print(f"  Wrote MD visualization: {out_path}")




# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR ENTRY
# ---------------------------------------------------------------------------
def main(mode: str = "primary") -> None:
    """
    mode = "primary"  → OF P1 vs Step7 Baseline
    mode = "advanced" → OF FINAL vs Step7 RETRO
    """

    print("\n==============================")
    print("  ORCHESTRATOR — FINAL BUILD")
    print("==============================\n")

    # ----------------------------------------------------------------------
    # 1) Resolve packet pair based on mode
    # ----------------------------------------------------------------------
    if mode == "primary":
        print("Mode: PRIMARY (OF P1 vs Step7 Baseline)\n")
        tag_a = ("OF", "p1")
        tag_b = ("STEP7", "baseline")

    elif mode == "advanced":
        print("Mode: ADVANCED (OF FINAL vs Step7 RETRO)\n")
        tag_a = ("OF", "final")
        tag_b = ("STEP7", "retro")

    else:
        raise ValueError("Mode must be 'primary' or 'advanced'")

    # Resolve pointers
    packet_a = resolve_latest_pointer(*tag_a)
    packet_b = resolve_latest_pointer(*tag_b)

    print("Resolved packets:")
    print(f"  A = {packet_a}")
    print(f"  B = {packet_b}\n")

    # ----------------------------------------------------------------------
    # Sanity-check summaries BEFORE running scorecard
    # ----------------------------------------------------------------------
    summarize_packet("A (OF)", packet_a)
    summarize_packet("B (STEP7)", packet_b)


    # ----------------------------------------------------------------------
    # 2) Create timestamped comparison output directory
    # ----------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = COMPARE_ROOT / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}\n")
    
    
    # ----------------------------------------------------------------------
    # Integrity checks BEFORE running scorecard
    # ----------------------------------------------------------------------
    ok_a = check_packet_integrity("A (OF)", packet_a)
    ok_b = check_packet_integrity("B (STEP7)", packet_b)

    if not (ok_a and ok_b):
        print("\n✗ Integrity check failed. Aborting comparison.\n")
        return

    # ----------------------------------------------------------------------
    # MD Visualization BEFORE running scorecard
    # ----------------------------------------------------------------------
    print("\n=== Generating MD Visualization ===")
    plot_md_comparison(packet_a, packet_b, out_dir)


    # ----------------------------------------------------------------------
    # 3) Run scorecard
    # ----------------------------------------------------------------------
    run_scorecard(packet_a, packet_b, out_dir)

    print("\n✓ Orchestrator complete.\n")





# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Default = primary mode
    # You can run advanced mode via:
    #     python orchestrator.py advanced
    mode = sys.argv[1] if len(sys.argv) > 1 else "primary"
    main(mode)

#!/usr/bin/env python3
"""
ML_OpticalFlow_Data_Collection_Driver.py

ML Data Collection Driver
Runs optical_flow_twoPass.py repeatedly with different configs,
auto-confirming prompts via --auto-yes.

UPDATED for bifurcated pipeline:

- Uses FAST SWEEP mode: --use-cached-of
- Assumes PRECOMPUTE-OF has already been run once:
    python optical_flow_twoPass.py --precompute-of --auto-yes
"""

import json
import subprocess
from pathlib import Path
import os
import re

# ---------------------------------------------------------
# CONFIG AREA (in plain view)
# ---------------------------------------------------------

OF_SCRIPT = Path(__file__).parent / "optical_flow_twoPass.py"

# Where to store ML training rows (JSONL)
DATA_OUT = Path("../outputs/OF_HO/ML_data_collection/ml_training_data.json")
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

# Parameter search space
PARAM_GRID = {
    "MOTION_THRESHOLD": [1.6, 1.8, 2.0, 2.2],
    "MIN_MOTION_PIXELS": [80, 120, 160],
    "MERGE_GAP_FRAMES": [12, 18, 24],
}

# ---------------------------------------------------------
# Helper: run OF in FAST SWEEP mode with overrides
# ---------------------------------------------------------

def run_of_with_overrides(override_dict: dict):
    """
    Run OF script in FAST SWEEP mode with a JSON override dict and auto-confirm enabled.

    This calls:
        python -X utf8 -u optical_flow_twoPass.py
            --use-cached-of
            --overrides '{"..."}'
            --auto-yes
    """

    override_json = json.dumps(override_dict)

    cmd = [
        "python",
        "-X", "utf8",
        "-u",                      # unbuffered stdout
        str(OF_SCRIPT),
        "--use-cached-of",         # FAST SWEEP mode (segmentation-only)
        "--overrides", override_json,
        "--auto-yes",
    ]

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"

    print(f"\n[RUN] {cmd}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        bufsize=1,
        env=env,
    )

    out_lines = []
    for line in proc.stdout:
        print(line, end="")         # live terminal output
        out_lines.append(line)

    proc.wait()
    stdout_text = "".join(out_lines)
    return stdout_text, ""          # stderr already merged


# ---------------------------------------------------------
# Helper: extract score from FAST SWEEP output
# ---------------------------------------------------------

import re

def extract_score(stdout_text: str):
    """
    Supports BOTH score formats:
      1) FINAL SCORE: 18.0 | TH=... PIX=... GAP=...
      2) [RESULT] score=201.0
    """
    for raw in stdout_text.splitlines():
        line = raw.strip()

        # New FAST SWEEP format
        m = re.search(r"\[RESULT\]\s*score=([-+]?\d*\.?\d+)", line)
        if m:
            return float(m.group(1))

        # Old full-run format
        m = re.search(r"FINAL SCORE:\s*([-+]?\d*\.?\d+)", line)
        if m:
            return float(m.group(1))

    return None




# ---------------------------------------------------------
# Main: iterate over parameter combinations
# ---------------------------------------------------------

def main():
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Simple grid search over PARAM_GRID
    for th in PARAM_GRID["MOTION_THRESHOLD"]:
        for pix in PARAM_GRID["MIN_MOTION_PIXELS"]:
            for mg in PARAM_GRID["MERGE_GAP_FRAMES"]:

                overrides = {
                    "MOTION_THRESHOLD": th,
                    "MIN_MOTION_PIXELS": pix,
                    "MERGE_GAP_FRAMES": mg,
                }

                stdout, stderr = run_of_with_overrides(overrides)
                score = extract_score(stdout)

                if score is None:
                    print("----- STDOUT (tail) -----")
                    print(stdout[-2000:])
                    print("----- STDERR (tail) -----")
                    print(stderr[-2000:])

                row = {
                    "overrides": overrides,
                    "score": score,
                    "stdout": stdout,
                }

                with DATA_OUT.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

                print(f"[RESULT] score={score} overrides={overrides}")


if __name__ == "__main__":
    main()

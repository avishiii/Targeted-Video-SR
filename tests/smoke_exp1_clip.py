"""CPU check of the Experiment 1 clip path: an mp4 named like the Pexels file,
target box from configs/boxes_exp1.json, MIL tracking, both degradations.

  python tests/smoke_exp1_clip.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "_smoke_exp1"


def make_clip(path, T=40, H=720, W=1280):
    rng = np.random.default_rng(1)
    bg = cv2.GaussianBlur(rng.integers(60, 200, (H, W, 3), dtype=np.uint8), (0, 0), 5)
    sign = np.full((180, 180, 3), 230, np.uint8)
    cv2.putText(sign, "BAD", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (60, 60, 140), 4)
    cv2.putText(sign, "VIBES", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60, 60, 140), 4)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for t in range(T):
        f = bg.copy()
        x, y = 600 - 2 * t, 60 + int(6 * np.sin(t / 5))
        f[y:y + 180, x:x + 180] = sign
        vw.write(f)
    vw.release()


def run(*cmd):
    print("$", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / "raw").mkdir(parents=True)
    make_clip(TMP / "raw" / "6655841-hd_1920_1080_30fps (3).mp4")
    run("scripts/build_dataset.py", "--preset", "folder", "--name", "exp1", "--root",
        str(TMP / "raw"), "--out", str(TMP / "tubes"), "--targets", "box_track",
        "--boxes-json", "configs/boxes_exp1.json", "--max-frames", "212",
        "--max-windows", "8", "--clip-len", "16", "--min-len", "8")
    run("scripts/run_baseline.py", "--data", str(TMP / "tubes/test"), "--methods", "bicubic",
        "--out", str(TMP / "results"))
    run("scripts/evaluate.py", "--data", str(TMP / "tubes/test"), "--results",
        str(TMP / "results"), "--no-lpips")
    metas = [json.loads(p.read_text()) for p in (TMP / "tubes/test").glob("*/meta.json")]
    assert metas, "no tubes built"
    m = metas[0]
    print("clip id:", m["clip"], "| tubes:", len(metas), "| HR", m["hr_size"], "LR", m["lr_size"],
          "| crop x0 first frames:", m["plan"]["x0"][:4])
    print("SMOKE TEST PASSED")

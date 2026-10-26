"""CPU-only end-to-end check: fake DAVIS -> tubes -> bicubic -> metrics.

  python tests/smoke_test.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "_smoke"


def make_fake_davis(root, videos, T=24, H=240, W=320, side=110):
    rng = np.random.default_rng(0)
    for v in videos:
        bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        tex = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
        tex = cv2.resize(cv2.resize(tex, (side // 4, side // 4)), (side, side), cv2.INTER_NEAREST)
        for t in range(T):
            x, y = 20 + 6 * t, 40 + int(20 * np.sin(t / 4))
            img = bg.copy()
            img[y:y + side, x:x + side] = tex
            m = np.zeros((H, W), np.uint8)
            m[y:y + side, x:x + side] = 1
            (root / "JPEGImages/480p" / v).mkdir(parents=True, exist_ok=True)
            (root / "Annotations/480p" / v).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "JPEGImages/480p" / v / f"{t:05d}.jpg"), img)
            p = Image.fromarray(m, mode="P")
            p.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
            p.save(root / "Annotations/480p" / v / f"{t:05d}.png")
    sets = root / "ImageSets/2017"
    sets.mkdir(parents=True, exist_ok=True)
    (sets / "train.txt").write_text(videos[0] + "\n")
    (sets / "val.txt").write_text("\n".join(videos[1:]) + "\n")


def run(*cmd):
    print("$", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    shutil.rmtree(TMP, ignore_errors=True)
    make_fake_davis(TMP / "DAVIS", ["aaa", "bbb", "ccc"])
    run("scripts/build_dataset.py", "--preset", "davis", "--root", str(TMP / "DAVIS"),
        "--out", str(TMP / "tubes"), "--clip-len", "16", "--min-len", "8")
    run("scripts/run_baseline.py", "--data", str(TMP / "tubes/test"), "--methods", "bicubic",
        "--out", str(TMP / "results"))
    run("scripts/evaluate.py", "--data", str(TMP / "tubes/test"), "--results",
        str(TMP / "results"), "--no-lpips")
    meta = json.loads(next((TMP / "tubes/test").glob("*/meta.json")).read_text())
    print("tube HR", meta["hr_size"], "LR", meta["lr_size"])
    assert meta["hr_size"][0] == 4 * meta["lr_size"][0]
    print("SMOKE TEST PASSED")

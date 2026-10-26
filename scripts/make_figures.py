"""Qualitative figures for the report/slides from results/ folders.

  python scripts/make_figures.py --data data/tubes/test --results results \
      --methods bicubic realesrgan sd_x4 ours_svd --out figures

Per sample:
  <id>_frames.png    middle frame: LR | each method | GT
  <id>_xt.png        temporal profile: one pixel row stacked over time.
                     Straight, smooth streaks = temporally stable; jagged,
                     noisy streaks = flicker. The standard VSR flicker figure.
Plus figures/runtime.png: seconds per frame per method (from _runtime.jsonl).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr.io import read_dir  # noqa: E402

PANEL = 256


def tag(img, text):
    img = cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA if img.shape[0] > PANEL
                     else cv2.INTER_NEAREST)
    out = np.full((PANEL + 24, PANEL, 3), 255, np.uint8)
    out[24:] = img
    cv2.putText(out, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def xt_slice(frames, row):
    return np.stack([f[row] for f in frames])  # (T, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--filter", default="")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)

    samples = sorted(p for p in Path(args.data).iterdir()
                     if (p / "meta.json").exists() and args.filter in p.name)
    done = 0
    for s in samples:
        if not all((res / m / s.name).exists() for m in args.methods):
            continue
        gt = read_dir(s / "hr")
        lr = read_dir(s / "lr")
        mid = len(gt) // 2
        seqs = {m: read_dir(res / m / s.name) for m in args.methods}
        panels = [tag(lr[mid], "LR input")] + [tag(seqs[m][mid], m) for m in args.methods] \
            + [tag(gt[mid], "ground truth")]
        cv2.imwrite(str(out / f"{s.name}_frames.png"), cv2.cvtColor(np.hstack(panels),
                                                                   cv2.COLOR_RGB2BGR))
        mask = read_dir(s / "mask", grayscale=True)[mid] > 127
        row = int(np.nonzero(mask.any(1))[0].mean()) if mask.any() else gt[0].shape[0] // 2
        cols = [("ground truth", gt)] + [(m, seqs[m]) for m in args.methods]
        strips = []
        for name, fr in cols:
            sl = cv2.resize(xt_slice(fr, row), (PANEL, max(48, 4 * len(fr))),
                            interpolation=cv2.INTER_NEAREST)
            head = np.full((22, PANEL, 3), 255, np.uint8)
            cv2.putText(head, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            strips.append(np.vstack([head, sl]))
        cv2.imwrite(str(out / f"{s.name}_xt.png"), cv2.cvtColor(np.hstack(strips),
                                                               cv2.COLOR_RGB2BGR))
        done += 1
        if done >= args.limit:
            break
    print(f"wrote figures for {done} samples to {out}")

    rt = {}
    for m in args.methods:
        rows = [json.loads(line) for f in (res / m).glob("_runtime*.jsonl")
                for line in f.read_text().splitlines() if line.strip()]
        if rows:
            rt[m] = (np.mean([r["sec_per_frame"] for r in rows]),
                     max((r["peak_gb"] or 0) for r in rows))
    if rt:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(list(rt), [v[0] for v in rt.values()], color="#c98a2b")
            for i, (sec, gb) in enumerate(rt.values()):
                ax.text(i, sec, f"{sec:.2f}s\n{gb:.1f} GB", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("seconds / frame")
            ax.set_title("Runtime on the target tube")
            fig.tight_layout()
            fig.savefig(out / "runtime.png", dpi=160)
        except ImportError:
            pass
        (out / "runtime.json").write_text(json.dumps(rt, indent=1))


if __name__ == "__main__":
    main()

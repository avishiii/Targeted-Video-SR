"""Score every method folder in results/ against the HR tubes.

  python scripts/evaluate.py --data data/tubes/test --results results

Writes results/metrics_per_sample.csv and results/summary.md (one table per
dataset x degradation, mean over samples). GT flow is computed once per sample
and shared by all methods.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr import metrics  # noqa: E402
from tsr.io import read_dir  # noqa: E402

COLS = ["psnr", "ssim", "lpips", "psnr_mask", "ssim_mask", "lpips_mask", "tlpips", "warp"]
NR_COLS = ["clipiqa", "musiq"]  # no-reference, as in Experiment 1 (pyiqa)
HIGHER = {"psnr", "ssim", "psnr_mask", "ssim_mask", "clipiqa", "musiq"}


class NoRef:
    """CLIP-IQA and MUSIQ via pyiqa, the same metrics Experiment 1 reported."""

    def __init__(self):
        import pyiqa
        import torch
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = dev
        self.m = {n: pyiqa.create_metric(n, device=dev) for n in NR_COLS}

    def __call__(self, frames):
        out = {n: [] for n in self.m}
        with self.torch.no_grad():
            for f in frames:
                x = self.torch.from_numpy(f).permute(2, 0, 1)[None].float().div(255).to(self.dev)
                for n, fn in self.m.items():
                    out[n].append(float(fn(x).item()))
        return {n: float(np.mean(v)) for n, v in out.items()}


def _nanmean(v):
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", default=None, help="default: every folder")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--nr", action="store_true",
                    help="also CLIP-IQA and MUSIQ (pip install pyiqa); adds a GT row")
    args = ap.parse_args()

    res = Path(args.results)
    methods = args.methods or sorted(p.name for p in res.iterdir()
                                     if p.is_dir() and not p.name.startswith("_"))
    lp = None
    if not args.no_lpips:
        try:
            lp = metrics.LPIPS()
        except ImportError:
            print("lpips/torch not installed: skipping LPIPS and tLPIPS")

    nr = NoRef() if args.nr else None
    cols = COLS + (NR_COLS if nr else [])
    samples = sorted(p for p in Path(args.data).iterdir() if (p / "meta.json").exists())
    rows = []
    for s in tqdm(samples, desc="samples"):
        meta = json.loads((s / "meta.json").read_text())
        gt, masks = read_dir(s / "hr"), read_dir(s / "mask", grayscale=True)
        flows = [metrics.flow_pair(gt[t], gt[t - 1]) for t in range(1, len(gt))]
        base = {"sample_id": s.name, "dataset": meta["dataset"],
                "degradation": meta["degradation"]["preset"]}
        if nr:  # reference point: how the HR ground truth itself scores
            rows.append({"method": "_gt_hr", **base, **{c: np.nan for c in COLS + ["warp_gt"]},
                         **nr(gt)})
        for m in methods:
            d = res / m / s.name
            if not d.exists():
                continue
            sr = read_dir(d)
            if len(sr) != len(gt) or sr[0].shape != gt[0].shape:
                print(f"skip {m}/{s.name}: shape/length mismatch")
                continue
            r = metrics.evaluate_tube(sr, gt, masks, lp, flows)
            if nr:
                r.update(nr(sr))
            rows.append({"method": m, **base, **r})

    if not rows:
        raise SystemExit("No results found to evaluate.")
    with open(res / "metrics_per_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["dataset"], r["degradation"])][r["method"]].append(r)
    lines = ["# Region-only results (mean over tubes)\n",
             "Arrows: ↑ higher is better, ↓ lower is better. *_mask = inside the target mask. "
             "warp = warping error ×1e3 on GT flow; GT's own warp error is the floor.\n"]
    for (ds, deg), by_m in sorted(groups.items()):
        n = len(next(iter(by_m.values())))
        lines.append(f"\n## {ds} · {deg} degradation · {n} tubes\n")
        lines.append("| method | " + " | ".join(f"{c} {'↑' if c in HIGHER else '↓'}" for c in cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        means = {m: {c: _nanmean([r[c] for r in rs]) for c in cols + ["warp_gt"]}
                 for m, rs in by_m.items()}
        for m in sorted(means):
            cells = ["–" if np.isnan(means[m][c]) else f"{means[m][c]:.4f}" if "ssim" in c
                     or "lpips" in c or c == "clipiqa" else f"{means[m][c]:.2f}" for c in cols]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        floor = _nanmean([v["warp_gt"] for v in means.values()])
        lines.append(f"\nGT warp-error floor: {floor:.2f}\n")
    (res / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

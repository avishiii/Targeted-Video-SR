"""Analyse DAVIS 2017 annotations and pick the evaluation clips, before any GPU work.

Reads ONLY the annotation PNGs (fast, CPU, ~1 min for all of DAVIS) and answers:
  * how big are the targets, how fast do they move, how long are they visible?
  * which objects survive our tube rules (min/max side, margin, window length)?
  * which are genuinely *targeted* (tube much smaller than the frame) and not
    just "the whole frame"?
  * how long will the diffusion runs take for the chosen subset?

  python scripts/analyze_dataset.py --root data/raw/DAVIS --out analysis \
      --n-test 8 --n-val 4 --sec-per-frame ours_svd=4.0 sd_x4=1.5

Writes analysis/objects.csv, summary.md, distributions.png and the clip lists
analysis/selected_test.txt, selected_val.txt (lines "video:object") that
build_dataset.py --clip-list consumes. Selection uses only annotation
statistics, never results, so it cannot bias the comparison between methods.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr import tubes  # noqa: E402


def object_stats(ann_dir, clip_len, margin, min_side, max_side):
    files = sorted(Path(ann_dir).glob("*.png"))
    masks = [np.array(Image.open(f)) for f in files]
    H, W = masks[0].shape[:2]
    ids = sorted({int(v) for m in masks for v in np.unique(m) if v not in (0, 255)})
    rows = []
    for oid in ids:
        boxes = [tubes.mask_to_box(m == oid) if (m == oid).sum() >= 64 else None for m in masks]
        vis = [b for b in boxes if b is not None]
        if not vis:
            continue
        b = np.array(vis, float)
        side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (H * W)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        speed = np.hypot(np.diff(cx), np.diff(cy)) if len(b) > 1 else np.array([0.0])
        filled = tubes.fill_gaps(boxes, 3)
        runs = tubes.contiguous_runs(filled)
        longest = max((e - s for s, e in runs), default=0)
        status, tube_side, tube_frac, factor = "no_window", 0, np.nan, np.nan
        for s, e in runs:
            if e - s >= clip_len:
                plan = tubes.plan_window(filled[s:s + clip_len], (H, W), margin=margin,
                                         min_side=min_side, max_side=max_side)
                if plan is None:
                    status = "too_small"
                else:
                    status, tube_side = "ok", plan["side_raw"]
                    tube_frac, factor = plan["side_raw"] / min(H, W), plan["factor"]
                break
        rows.append(dict(video=Path(ann_dir).name, obj=oid, frames=len(masks), H=H, W=W,
                         visible=round(len(vis) / len(masks), 3), longest_run=longest,
                         box_side_med=round(float(np.median(side)), 1),
                         box_side_max=round(float(side.max()), 1),
                         area_frac_med=round(float(np.median(area)), 4),
                         speed_med=round(float(np.median(speed)), 2),
                         speed_p90=round(float(np.percentile(speed, 90)), 2),
                         status=status, tube_side=tube_side,
                         tube_frac=round(float(tube_frac), 3) if tube_side else "",
                         downscale=round(float(factor), 3) if tube_side else ""))
    return rows


def pick(rows, n, max_tube_frac, exclude_videos=()):
    """One object per video, eligible = ok + genuinely targeted; spread over motion."""
    cand = [r for r in rows if r["status"] == "ok" and r["tube_frac"] != ""
            and r["tube_frac"] <= max_tube_frac and r["video"] not in exclude_videos]
    best = {}
    for r in cand:  # per video keep the most "targeted" object (smallest tube)
        if r["video"] not in best or r["tube_frac"] < best[r["video"]]["tube_frac"]:
            best[r["video"]] = r
    pool = sorted(best.values(), key=lambda r: r["speed_med"])
    if len(pool) <= n:
        return pool
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)  # slow ... fast
    return [pool[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DAVIS root (has Annotations/480p)")
    ap.add_argument("--res", default="480p")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--clip-len", type=int, default=27)
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--min-side", type=int, default=128)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--max-tube-frac", type=float, default=0.75,
                    help="tube side / frame short side; above this it is not 'targeted'")
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--degradations", type=int, default=2)
    ap.add_argument("--sec-per-frame", nargs="*", default=[],
                    help="method=seconds from _runtime.jsonl, for the time budget")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = root / "ImageSets" / "2017"
    split_of = {v: "train" for v in (sets / "train.txt").read_text().split()}
    split_of.update({v: "val" for v in (sets / "val.txt").read_text().split()})

    rows = []
    for d in sorted((root / "Annotations" / args.res).iterdir()):
        for r in object_stats(d, args.clip_len, args.margin, args.min_side, args.max_side):
            r["split"] = split_of.get(r["video"], "?")
            rows.append(r)
    with open(out / "objects.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # DAVIS val is the standard evaluation set; we never train, so train is unused.
    val_rows = [r for r in rows if r["split"] == "val"]
    test = pick(val_rows, args.n_test, args.max_tube_frac)
    val = pick(val_rows, args.n_val, args.max_tube_frac, {r["video"] for r in test})
    for name, sel in (("test", test), ("val", val)):
        (out / f"selected_{name}.txt").write_text(
            "".join(f"{r['video']}:{r['obj']}\n" for r in sel))

    def q(key, rs):
        v = np.array([r[key] for r in rs], float)
        return " / ".join(f"{x:.3g}" for x in np.percentile(v, [10, 50, 90])) if len(v) else "–"

    L = ["# DAVIS 2017 analysis\n",
         f"Tube rules: window {args.clip_len} frames, margin {args.margin}, "
         f"HR side {args.min_side}–{args.max_side} px, 'targeted' = tube ≤ "
         f"{args.max_tube_frac:.0%} of the frame's short side.\n",
         "| split | videos | objects | ok | too small | no full window | ok & targeted |",
         "|---|---|---|---|---|---|---|"]
    for sp in ("train", "val"):
        rs = [r for r in rows if r["split"] == sp]
        ok = [r for r in rs if r["status"] == "ok"]
        tg = [r for r in ok if r["tube_frac"] <= args.max_tube_frac]
        L.append(f"| {sp} | {len({r['video'] for r in rs})} | {len(rs)} | {len(ok)} | "
                 f"{sum(r['status'] == 'too_small' for r in rs)} | "
                 f"{sum(r['status'] == 'no_window' for r in rs)} | {len(tg)} |")
    L += ["\nPercentiles 10 / 50 / 90 (val objects):\n",
          f"* frames per video: {q('frames', val_rows)}",
          f"* target box long side (px @480p): {q('box_side_med', val_rows)}",
          f"* target area (fraction of frame): {q('area_frac_med', val_rows)}",
          f"* motion (px/frame, median per object): {q('speed_med', val_rows)}",
          f"* visible fraction: {q('visible', val_rows)}",
          "\n## Selected (one object per video, spread from slow to fast motion)\n",
          "| set | video:obj | box side | tube side | tube/frame | speed px/f |",
          "|---|---|---|---|---|---|"]
    for name, sel in (("test", test), ("val", val)):
        for r in sel:
            L.append(f"| {name} | {r['video']}:{r['obj']} | {r['box_side_med']} | "
                     f"{r['tube_side']} | {r['tube_frac']} | {r['speed_med']} |")

    n_tubes = (len(test) + 1) * args.degradations  # +1 = Exp 1 sign clip window
    frames = n_tubes * args.clip_len
    L += [f"\n## Time budget (test set: {len(test)} DAVIS + 1 Exp 1 window × "
          f"{args.degradations} degradations = {n_tubes} tubes, {frames} frames)\n"]
    for spec in args.sec_per_frame:
        m, s = spec.split("=")
        L.append(f"* {m}: {frames * float(s) / 3600:.2f} h on one GPU "
                 f"({frames * float(s) / 7200:.2f} h split over 2 GPUs)")
    if not args.sec_per_frame:
        L.append("* run `run_baseline.py --limit 1` per method, then pass "
                 "`--sec-per-frame method=sec` to fill this in.")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        for a, key, title in zip(ax, ("box_side_med", "speed_med", "tube_frac"),
                                 ("target box long side (px)", "motion (px/frame)",
                                  "tube side / frame short side")):
            vals = [r[key] for r in val_rows if r[key] != ""]
            a.hist(vals, bins=15, color="#c98a2b")
            sel_vals = [r[key] for r in test if r[key] != ""]
            a.plot(sel_vals, [0] * len(sel_vals), "v", color="#13202f", ms=8, label="selected")
            a.set_title(title, fontsize=10)
        ax[2].axvline(args.max_tube_frac, ls="--", color="gray")
        ax[0].legend(fontsize=8)
        fig.suptitle("DAVIS 2017 val objects", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "distributions.png", dpi=160)
    except ImportError:
        pass
    (out / "selection.json").write_text(json.dumps({"test": test, "val": val}, indent=1))


if __name__ == "__main__":
    main()

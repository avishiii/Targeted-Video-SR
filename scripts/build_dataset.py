"""Build paired (LR tube, HR tube, mask) samples.

Examples
  # DAVIS 2017 with its ground-truth masks (no SAM 2 needed)
  python scripts/build_dataset.py --preset davis --root data/raw/DAVIS --out data/tubes

  # REDS with SAM 2 tracking objects found by YOLO on frame 0 (REDS4 = test)
  python scripts/build_dataset.py --preset reds --root data/raw/REDS --out data/tubes \
      --targets sam2_detect

  # CelebV-HQ (or any folder of clips) where the whole frame is the target
  python scripts/build_dataset.py --preset folder --name celebv --root data/raw/celebv \
      --targets whole --margin 1.0 --auto-split

Output: <out>/<split>/<sample_id>/{hr,lr,mask}/NNNNN.png + meta.json,
        and <out>/<split>.jsonl indexing the samples.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr import degrade, targets, tubes  # noqa: E402
from tsr.io import VID_EXT, read_frames, write_frames  # noqa: E402

REDS4 = {"000", "011", "015", "020"}  # standard REDS test clips (taken from the train set)


def hash_split(name, val=0.1, test=0.1):
    u = (zlib.crc32(name.encode()) % 10_000) / 10_000
    return "test" if u < test else "val" if u < test + val else "train"


def list_clips(args):
    root = Path(args.root)
    clips = []
    if args.preset == "davis":
        sets = root / "ImageSets" / "2017"
        train = (sets / "train.txt").read_text().split()
        val = sorted((sets / "val.txt").read_text().split())
        for v in train:
            clips.append(("train", v))
        for i, v in enumerate(val):  # DAVIS has no public test GT: halve val
            clips.append(("val" if i % 2 == 0 else "test", v))
        return [dict(dataset="davis", clip=v, split=s,
                     frames=root / "JPEGImages" / "480p" / v,
                     masks=root / "Annotations" / "480p" / v) for s, v in clips]

    if args.preset == "reds":
        def find(name):
            hits = [p for p in root.rglob(name) if p.is_dir()]
            return hits[0] if hits else None
        tr, va = find("train_sharp"), find("val_sharp")
        out = []
        if tr:
            for d in sorted(p for p in tr.iterdir() if p.is_dir()):
                out.append(dict(dataset="reds", clip=d.name, frames=d,
                                split="test" if d.name in REDS4 else "train"))
        if va:
            for d in sorted(p for p in va.iterdir() if p.is_dir()):
                out.append(dict(dataset="reds", clip=f"val{d.name}", frames=d, split="val"))
        if not out:
            raise SystemExit(f"No train_sharp/ or val_sharp/ under {root}")
        return out

    # generic folder: each subfolder of frames or video file is one clip
    items = sorted(p for p in root.iterdir() if p.is_dir() or p.suffix.lower() in VID_EXT)
    return [dict(dataset=args.name, clip=targets.safe_id(p.stem), frames=p,
                 split=hash_split(p.stem) if args.auto_split else args.split) for p in items]


def get_tracks(args, clip, frames, names, tracker):
    if args.targets == "masks":
        mdir = clip.get("masks") or Path(args.masks_root) / clip["clip"]
        return targets.masks_from_annotations(mdir, names)
    if args.targets == "whole":
        return targets.whole_frame_target(frames)
    if args.targets == "box_track":
        return targets.track_boxes_mil(frames, targets.boxes_from_json(args.boxes_json, clip["clip"]))
    if args.targets == "sam2_boxes":
        boxes0 = targets.boxes_from_json(args.boxes_json, clip["clip"])
    else:  # sam2_detect
        boxes0 = targets.detect_boxes(frames[0])
    return tracker.track(frames, boxes0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["davis", "reds", "folder"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/tubes")
    ap.add_argument("--name", default="custom", help="dataset name for --preset folder")
    ap.add_argument("--split", default="test", help="split for --preset folder")
    ap.add_argument("--auto-split", action="store_true", help="80/10/10 split by clip name")
    ap.add_argument("--only-splits", nargs="*", default=None, help="e.g. --only-splits val test")
    ap.add_argument("--clip-list", default=None,
                    help='lines "video" or "video:obj" (from analyze_dataset.py); others skipped')
    ap.add_argument("--list-split", default=None,
                    help="split name for --clip-list clips (e.g. test); default keeps preset split")
    ap.add_argument("--targets", default=None,
                    choices=["masks", "box_track", "sam2_boxes", "sam2_detect", "whole"])
    ap.add_argument("--masks-root", default=None)
    ap.add_argument("--boxes-json", default=None)
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--degradation", nargs="+", default=["realworld", "bicubic"],
                    choices=list(degrade.PRESETS))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--min-side", type=int, default=128, help="HR px; LR = /scale")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--clip-len", type=int, default=32, help="frames per tube window")
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=2, help="per target track")
    ap.add_argument("--max-frames", type=int, default=100, help="frames read per clip")
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=643)
    args = ap.parse_args()

    if args.targets is None:
        args.targets = "masks" if args.preset == "davis" else "whole"
    tracker = None
    if args.targets.startswith("sam2"):
        tracker = targets.SAM2Tracker(args.sam2_model)

    clips = list_clips(args)
    allowed = None
    if args.clip_list:
        allowed = {}
        for line in Path(args.clip_list).read_text().split():
            v, _, o = line.partition(":")
            allowed.setdefault(v, set())
            if o:
                allowed[v].add(int(o))
        clips = [c for c in clips if c["clip"] in allowed]
        if args.list_split:
            for c in clips:
                c["split"] = args.list_split
    if args.only_splits:
        clips = [c for c in clips if c["split"] in args.only_splits]
    clips = clips[: args.max_clips]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {"clips": 0, "samples": 0, "too_small": 0, "no_target": 0}

    for clip in tqdm(clips, desc="clips"):
        frames, names = read_frames(clip["frames"], max_frames=args.max_frames)
        if len(frames) < args.min_len:
            continue
        stats["clips"] += 1
        tracks = get_tracks(args, clip, frames, names, tracker)
        if allowed and allowed.get(clip["clip"]):
            tracks = {o: m for o, m in tracks.items() if o in allowed[clip["clip"]]}
        if not tracks:
            stats["no_target"] += 1
        for oid, masks in tracks.items():
            boxes = tubes.fill_gaps([tubes.mask_to_box(m) for m in masks], args.max_gap)
            for run in tubes.contiguous_runs(boxes):
                for w, (s, e) in enumerate(tubes.windows(*run, args.clip_len, args.min_len,
                                                         args.max_windows)):
                    plan = tubes.plan_window(boxes[s:e], frames[0].shape[:2], scale=args.scale,
                                             margin=args.margin, min_side=args.min_side,
                                             max_side=args.max_side)
                    if plan is None:
                        stats["too_small"] += 1
                        continue
                    hr_pad = [tubes.extract_padded(frames[s + i], plan, i) for i in range(e - s)]
                    hr = [tubes.unpad(f, plan["pad"]) for f in hr_pad]
                    mask = [tubes.extract_mask(masks[s + i] if masks[s + i] is not None
                                               else np.zeros(frames[0].shape[:2], bool), plan, i)
                            for i in range(e - s)]
                    for preset in args.degradation:
                        sid = f"{clip['dataset']}_{clip['clip']}_o{oid}_w{w}_{preset}"
                        rng = np.random.default_rng(args.seed + zlib.crc32(sid.encode()))
                        params = degrade.sample_params(rng, preset)
                        lr_pad = degrade.degrade_clip(hr_pad, params, args.scale, rng)
                        lr = [tubes.unpad(f, plan["pad"] // args.scale) for f in lr_pad]
                        d = out_root / clip["split"] / sid
                        fnames = names[s:e]
                        write_frames(d / "hr", hr, fnames)
                        write_frames(d / "lr", lr, fnames)
                        write_frames(d / "mask", [m * 255 for m in mask], fnames)
                        meta = dict(sample_id=sid, dataset=clip["dataset"], clip=clip["clip"],
                                    obj=oid, split=clip["split"], frames=fnames, start=s,
                                    degradation=params, plan=plan, scale=args.scale,
                                    hr_size=hr[0].shape[:2], lr_size=lr[0].shape[:2])
                        (d / "meta.json").write_text(json.dumps(meta, indent=1))
                        stats["samples"] += 1
    write_index(out_root)
    print(json.dumps(stats, indent=1))


def write_index(out_root):
    """Rebuild <split>.jsonl from what is on disk (safe to re-run the builder)."""
    for split_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        rows = []
        for m in sorted(split_dir.glob("*/meta.json")):
            meta = json.loads(m.read_text())
            rows.append({"sample_id": meta["sample_id"], "path": str(m.parent),
                         "dataset": meta["dataset"],
                         "degradation": meta["degradation"]["preset"]})
        (out_root / f"{split_dir.name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()

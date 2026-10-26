"""Run baselines on every tube of a split. Resumable: finished samples are skipped,
so a Colab disconnect costs at most one sample.

  python scripts/run_baseline.py --data data/tubes/test --methods bicubic realesrgan sd_x4
"""
import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsr import baselines  # noqa: E402
from tsr.io import image_files, read_dir, write_frames  # noqa: E402


def peak_mem_gb(reset=False):
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # no torch / broken install: bicubic still runs
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="split folder, e.g. data/tubes/test")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--filter", default="", help="only samples whose id contains this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1",
                    help="i/n: process every n-th sample from i (2 GPUs: 0/2 and 1/2)")
    ap.add_argument("--sd-steps", type=int, default=30)
    ap.add_argument("--sd-noise", type=int, default=60, help="Exp 1 targeted used 60")
    ap.add_argument("--sd-guidance", type=float, default=1.0)
    ap.add_argument("--sd-offload", action="store_true", help="CPU offload, as in Exp 1")
    ap.add_argument("--svd-strength", type=float, default=0.5,
                    help="fraction of the noise schedule re-run (0 = copy input, 1 = full generation)")
    ap.add_argument("--svd-steps", type=int, default=25)
    ap.add_argument("--svd-work-size", type=int, default=512)
    ap.add_argument("--svd-fix-sigma", type=float, default=3.0, help="0 disables the low-freq fix")
    ap.add_argument("--svd-offload", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. _s0.3")
    args = ap.parse_args()

    samples = sorted(p for p in Path(args.data).iterdir()
                     if (p / "meta.json").exists() and args.filter in p.name)[: args.limit]
    i, n = (int(v) for v in args.shard.split("/"))
    samples = samples[i::n]
    for name in args.methods:
        if name == "ours_svd":
            kw = dict(strength=args.svd_strength, steps=args.svd_steps,
                      work_size=args.svd_work_size, fix_sigma=args.svd_fix_sigma,
                      offload=args.svd_offload)
        else:
            kw = dict(steps=args.sd_steps, noise_level=args.sd_noise,
                      guidance=args.sd_guidance, offload=args.sd_offload)
        method = baselines.build(name, **kw)
        name = name + args.tag  # ablations get their own results folder
        log = Path(args.out) / name / f"_runtime{'' if n == 1 else f'_{i}'}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        for s in tqdm(samples, desc=name):
            lr_files = image_files(s / "lr")
            out_dir = Path(args.out) / name / s.name
            if out_dir.exists() and len(image_files(out_dir)) == len(lr_files):
                continue
            lr = read_dir(s / "lr")
            peak_mem_gb(reset=True)
            t0 = time.time()
            sr = method.process(lr)
            dt = time.time() - t0
            exp = (lr[0].shape[0] * 4, lr[0].shape[1] * 4)
            assert sr[0].shape[:2] == exp, f"{name}: got {sr[0].shape[:2]}, expected {exp}"
            write_frames(out_dir, sr, [f.stem for f in lr_files])
            with open(log, "a") as fh:
                fh.write(json.dumps({"sample_id": s.name, "frames": len(lr),
                                     "sec": round(dt, 2), "sec_per_frame": round(dt / len(lr), 3),
                                     "peak_gb": peak_mem_gb()}) + "\n")
        del method


if __name__ == "__main__":
    main()

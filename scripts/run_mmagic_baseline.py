"""BasicVSR++ and RealBasicVSR via MMagic, in a SEPARATE environment.

MMagic pins old mmcv/torch versions and conflicts with diffusers/SAM 2, so run
this in its own Colab runtime or conda env (see README). It only reads LR tube
folders and writes SR folders, so it shares nothing else with the main code.

  python scripts/run_mmagic_baseline.py --data data/tubes/test --model basicvsr_pp
  python scripts/run_mmagic_baseline.py --data data/tubes/test --model real_basicvsr

Which weights to use:
  basicvsr_pp   - check the printed config on first run. For the `bicubic`
                  degradation, the REDS4 BI checkpoint is the fair one. On
                  `realworld` tubes it is expected to look soft: it was never
                  trained on noise or compression. That is the point of including it.
  real_basicvsr - the real-world variant; compare it on `realworld` tubes.
"""
import argparse
import tempfile
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg"}


def frames_in(d):
    return sorted(f for f in Path(d).iterdir() if f.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["basicvsr_pp", "real_basicvsr"], required=True)
    ap.add_argument("--setting", type=int, default=None, help="MMagic model_setting index")
    ap.add_argument("--out", default="results")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    from mmagic.apis import MMagicInferencer
    kw = {"model_name": args.model}
    if args.setting is not None:
        kw["model_setting"] = args.setting
    editor = MMagicInferencer(**kw)
    print(editor.inferencer.cfg.get("model", "") if hasattr(editor, "inferencer") else "")

    samples = sorted(p for p in Path(args.data).iterdir()
                     if (p / "meta.json").exists() and args.filter in p.name)
    for s in samples:
        lr = frames_in(s / "lr")
        out_dir = Path(args.out) / args.model / s.name
        if out_dir.exists() and len(frames_in(out_dir)) == len(lr):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            editor.infer(video=str(s / "lr"), result_out_dir=tmp)
            produced = sorted(f for f in Path(tmp).rglob("*") if f.suffix.lower() in IMG_EXT)
            if len(produced) != len(lr):
                raise RuntimeError(f"{s.name}: MMagic wrote {len(produced)} frames, "
                                   f"expected {len(lr)}. Inspect {tmp} layout.")
            out_dir.mkdir(parents=True, exist_ok=True)
            for src, ref in zip(produced, lr):  # re-save as PNG named like the LR/HR frames
                cv2.imwrite(str(out_dir / f"{ref.stem}.png"), cv2.imread(str(src)))
        print("done", s.name)


if __name__ == "__main__":
    main()

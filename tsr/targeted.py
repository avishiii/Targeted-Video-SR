"""End-to-end targeted SR on a whole clip (what the demo runs).

  LR clip + frame-0 box -> track -> square tube -> restore tube with a method
  -> paste back into the baseline-upscaled frame with a feathered border.

The background is the SAME baseline used for comparison, so baseline and ours
differ only inside the target tube: any difference you see is the targeting.
"""
import subprocess

import cv2
import numpy as np

from . import degrade, metrics, tubes


def make_lr(frames, preset="bicubic", seed=0, scale=4):
    """HD clip -> (LR clip, GT clip). GT is cropped to a multiple of `scale`."""
    h, w = frames[0].shape[:2]
    gt = [f[: h - h % scale, : w - w % scale] for f in frames]
    rng = np.random.default_rng(seed)
    params = degrade.sample_params(rng, preset)
    if params.get("crf") is not None and (gt[0].shape[0] // scale) % 2:
        params["crf"] = None  # H.264 needs even LR sizes
    return degrade.degrade_clip(gt, params, scale, rng), gt


def feather_weights(n, frac=0.15):
    """n x n weights: 1 inside, smooth ramp to 0 over `frac` of the side at the border."""
    ramp = max(1, int(n * frac))
    d = np.minimum.reduce(np.meshgrid(np.arange(n), np.arange(n))
                          + np.meshgrid(n - 1 - np.arange(n), n - 1 - np.arange(n)))
    t = np.clip(d / ramp, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)[..., None]  # smoothstep


def plan_lr_tube(boxes_lr, lr_hw, margin=1.25, max_lr_side=128):
    plan = tubes.plan_window(boxes_lr, lr_hw, scale=1, margin=margin, align=8, min_side=16,
                             max_side=max_lr_side, pad=0)
    if plan is None:
        raise ValueError("Target too small: draw a box of at least ~16 LR pixels.")
    return plan


def restore_clip(lr, boxes_lr, method, baseline, scale=4, margin=1.25, max_lr_side=128,
                 feather=0.15):
    """Returns dict(ours, base, plan, tube_lr, tube_sr) with full-size frames."""
    plan = plan_lr_tube(boxes_lr, lr[0].shape[:2], margin, max_lr_side)
    tube_lr = [tubes.extract_padded(f, plan, i) for i, f in enumerate(lr)]
    tube_sr = method.process(tube_lr)
    base = baseline.process(lr)
    n = plan["side_raw"] * scale
    wgt = feather_weights(n, feather)
    ours = []
    for i, (b, t) in enumerate(zip(base, tube_sr)):
        if t.shape[0] != n:
            t = cv2.resize(t, (n, n), interpolation=cv2.INTER_CUBIC)
        x, y = plan["x0"][i] * scale, plan["y0"][i] * scale
        o = b.copy()
        region = o[y:y + n, x:x + n].astype(np.float32)
        o[y:y + n, x:x + n] = np.clip(t * wgt + region * (1 - wgt), 0, 255).astype(np.uint8)
        ours.append(o)
    return dict(ours=ours, base=base, plan=plan, tube_lr=tube_lr, tube_sr=tube_sr)


def tube_crops(frames, plan, scale=4):
    n = plan["side_raw"] * scale
    return [f[plan["y0"][i] * scale:plan["y0"][i] * scale + n,
              plan["x0"][i] * scale:plan["x0"][i] * scale + n] for i, f in enumerate(frames)]


def box_masks(boxes_lr, plan, scale=4):
    """Object-box masks inside each HR tube crop (uint8 0/255)."""
    n = plan["side_raw"] * scale
    out = []
    for i, (x0, y0, x1, y1) in enumerate(boxes_lr):
        m = np.zeros((n, n), np.uint8)
        ox, oy = plan["x0"][i], plan["y0"][i]
        a, b = int(max(0, (x0 - ox) * scale)), int(max(0, (y0 - oy) * scale))
        c, d = int(min(n, (x1 - ox) * scale)), int(min(n, (y1 - oy) * scale))
        m[b:d, a:c] = 255
        out.append(m)
    return out


def seam_score(frames, gt, plan, scale=4, band=3):
    """Mean |grad(out) - grad(gt)| in a thin band on the tube border (lower = less seam)."""
    n = plan["side_raw"] * scale
    vals = []
    for i, (f, g) in enumerate(zip(frames, gt)):
        x, y = plan["x0"][i] * scale, plan["y0"][i] * scale
        H, W = g.shape[:2]
        ring = np.zeros((H, W), bool)
        ring[max(0, y - band):y + n + band, max(0, x - band):x + n + band] = True
        ring[y + band:y + n - band, x + band:x + n - band] = False
        grads = []
        for img in (f, g):
            gr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            grads.append(np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0), cv2.Sobel(gr, cv2.CV_32F, 0, 1)))
        vals.append(float(np.abs(grads[0] - grads[1])[ring].mean()))
    return float(np.mean(vals))


def compare(result, gt, boxes_lr, lpips_model=None, scale=4):
    """Region metrics for baseline vs ours against GT (tube crops + object box)."""
    plan = result["plan"]
    gt_c = tube_crops(gt, plan, scale)
    masks = box_masks(boxes_lr, plan, scale)
    flows = [metrics.flow_pair(gt_c[t], gt_c[t - 1]) for t in range(1, len(gt_c))]
    rows = {}
    for name in ("base", "ours"):
        r = metrics.evaluate_tube(tube_crops(result[name], plan, scale), gt_c, masks,
                                  lpips_model, flows)
        r["seam"] = seam_score(result[name], gt, plan, scale)
        rows[name] = r
    return rows


def write_video(frames, path, fps=24):
    """H.264 mp4 that plays in browsers (cv2's mp4v often does not)."""
    h, w = frames[0].shape[:2]
    frames = [np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge") for f in frames]
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                   check=True)
    return str(path)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 12 + 11 * len(text)), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img

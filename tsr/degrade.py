"""Synthetic degradations that turn an HR tube into its LR input.

Two presets:
  bicubic   - antialiased ×4 bicubic only (the classic "BI" setting; fair to
              BasicVSR++, which was trained on it).
  realworld - Real-ESRGAN-style two-stage blur / resize / noise / JPEG with the
              op order shuffled per clip, plus optional H.264 compression.

Key design choice for video: every parameter (kernel, scale factors, noise
level, JPEG quality, CRF) is sampled ONCE per clip and reused for every frame.
Only the noise realisation changes per frame. Sampling per frame would create
artificial flicker in the LR input and unfairly punish temporal models.
"""
import os
import subprocess
import tempfile
import warnings

import cv2
import numpy as np
from PIL import Image

from .io import to_uint8

INTERPS = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

REALWORLD = dict(
    shuffle=True,
    blur_prob=1.0, blur_sigma1=(0.2, 2.5), blur_sigma2=(0.2, 1.2),
    aniso_prob=0.5, kernel_size=(7, 15),
    resize_range1=(0.5, 1.2),   # stage 1: relative to the HR size
    resize_range2=(0.75, 1.25), # stage 2: relative to the final LR size
    noise_prob=0.9, gauss_sigma=(1.0, 15.0), poisson_prob=0.4,
    poisson_scale=(0.05, 2.0), gray_noise_prob=0.4,
    jpeg_prob=0.9, jpeg_q=(40, 95),
    second_stage_prob=0.7,
    video_prob=0.5, crf=(18, 30),
)
# Lighter than the Real-ESRGAN paper on purpose: our LR targets are only
# 32-128 px, and the paper's settings can erase them completely.

PRESETS = {"realworld": REALWORLD, "bicubic": None}


# ----------------------------------------------------------------- sampling
def _sample_stage(rng, cfg, stage):
    p = {"order": ["blur", "resize", "noise", "jpeg"]}
    if rng.random() < cfg["blur_prob"]:
        lo, hi = cfg["kernel_size"]
        size = int(rng.integers(lo // 2, hi // 2 + 1)) * 2 + 1
        sig = cfg[f"blur_sigma{stage}"]
        sx = float(rng.uniform(*sig))
        sy = float(rng.uniform(*sig)) if rng.random() < cfg["aniso_prob"] else sx
        p["blur"] = [size, sx, sy, float(rng.uniform(0, np.pi))]
    p["resize"] = [float(rng.uniform(*cfg[f"resize_range{stage}"])),
                   str(rng.choice(list(INTERPS)))]
    if rng.random() < cfg["noise_prob"]:
        gray = bool(rng.random() < cfg["gray_noise_prob"])
        if rng.random() < cfg["poisson_prob"]:
            p["noise"] = ["poisson", float(rng.uniform(*cfg["poisson_scale"])), gray]
        else:
            p["noise"] = ["gauss", float(rng.uniform(*cfg["gauss_sigma"])), gray]
    if rng.random() < cfg["jpeg_prob"]:
        p["jpeg"] = int(rng.integers(cfg["jpeg_q"][0], cfg["jpeg_q"][1] + 1))
    if cfg["shuffle"]:
        rng.shuffle(p["order"])
    return p


def sample_params(rng, preset="realworld"):
    """Sample one clip's degradation. Returns a JSON-serialisable dict."""
    cfg = PRESETS[preset]
    if cfg is None:
        return {"preset": "bicubic", "stages": [], "final_interp": "pil_bicubic", "crf": None}
    stages = [_sample_stage(rng, cfg, 1)]
    if rng.random() < cfg["second_stage_prob"]:
        stages.append(_sample_stage(rng, cfg, 2))
    crf = int(rng.integers(cfg["crf"][0], cfg["crf"][1] + 1)) if rng.random() < cfg["video_prob"] else None
    return {"preset": preset, "stages": stages,
            "final_interp": str(rng.choice(list(INTERPS))), "crf": crf}


# ----------------------------------------------------------------- ops
def gaussian_kernel(size, sx, sy, theta):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def _add_noise(img, spec, rng):
    kind, level, gray = spec
    h, w = img.shape[:2]
    if kind == "gauss":
        shape = (h, w, 1) if gray else (h, w, 3)
        return img + rng.normal(0.0, level, shape).astype(np.float32)
    # Poisson (shot) noise: lower peak = stronger noise.
    peak = 255.0 / level
    src = img.mean(axis=2, keepdims=True) if gray else img
    src01 = np.clip(src / 255.0, 0, 1)
    noise = (rng.poisson(src01 * peak) / peak - src01) * 255.0
    return img + noise.astype(np.float32)


def _jpeg(img, quality):
    bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)


def _resize(img, wh, interp):
    wh = (max(8, int(round(wh[0]))), max(8, int(round(wh[1]))))
    return cv2.resize(img, wh, interpolation=INTERPS[interp])


def _apply_stage(img, p, stage_idx, lr_wh, rng):
    for op in p["order"]:
        if op == "blur" and "blur" in p:
            img = cv2.filter2D(img, -1, gaussian_kernel(*p["blur"]), borderType=cv2.BORDER_REFLECT)
        elif op == "resize":
            f, interp = p["resize"]
            h, w = img.shape[:2]
            target = (w * f, h * f) if stage_idx == 0 else (lr_wh[0] * f, lr_wh[1] * f)
            img = _resize(img, target, interp)
        elif op == "noise" and "noise" in p:
            img = _add_noise(img, p["noise"], rng)
        elif op == "jpeg" and "jpeg" in p:
            img = _jpeg(img, p["jpeg"])
    return np.clip(img, 0, 255)


def h264_roundtrip(frames, crf, preset="medium"):
    """Encode/decode frames with libx264 so the LR input has real codec artefacts."""
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        raise ValueError("H.264 (yuv420p) needs even frame sizes")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        enc = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-c:v", "libx264", "-preset", preset,
               "-crf", str(crf), "-pix_fmt", "yuv420p", path]
        subprocess.run(enc, input=b"".join(f.tobytes() for f in frames), check=True)
        dec = ["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(dec, capture_output=True, check=True).stdout
    out = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(out) != len(frames):
        raise RuntimeError(f"ffmpeg returned {len(out)} frames, expected {len(frames)}")
    return list(out.copy())


def degrade_clip(frames, params, scale, rng):
    """Degrade a list of HR uint8 frames into LR uint8 frames of size (H/scale, W/scale)."""
    h, w = frames[0].shape[:2]
    if h % scale or w % scale:
        raise ValueError(f"HR size {w}x{h} is not divisible by scale {scale}")
    lr_wh = (w // scale, h // scale)
    out = []
    for frame in frames:
        if params["final_interp"] == "pil_bicubic":
            # PIL bicubic is antialiased, close to MATLAB imresize (the BI protocol).
            out.append(np.asarray(Image.fromarray(frame).resize(lr_wh, Image.BICUBIC)))
            continue
        img = frame.astype(np.float32)
        for i, stage in enumerate(params["stages"]):
            img = _apply_stage(img, stage, i, lr_wh, rng)
        img = cv2.resize(img, lr_wh, interpolation=INTERPS[params["final_interp"]])
        out.append(to_uint8(img))

    if params.get("crf") is not None:
        try:
            out = h264_roundtrip(out, params["crf"])
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            warnings.warn(f"ffmpeg unavailable or failed ({e}); skipping video compression")
            params["crf"] = None  # record what was actually applied
    return out

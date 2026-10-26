"""Turn per-frame target masks into fixed-size, temporally smooth crop "tubes".

Rules that matter for fair evaluation:
  * The crop SIDE is constant over a window; only the centre moves. A per-frame
    size change would zoom in and out, which looks like flicker.
  * The HR ground truth is never upsampled. If the target is bigger than
    max_side we downscale the HR crop (keeps it sharp); if it is smaller than
    min_side we drop it (its LR version would be too small to be meaningful).
  * Side and padding are multiples of the SR scale (and of `align`), so the LR
    tube is an exact 1/scale of the HR tube and fits a latent-diffusion VAE
    (/8) at LR resolution.
"""
import math

import cv2
import numpy as np


def mask_to_box(mask):
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def fill_gaps(boxes, max_gap):
    """Linearly interpolate missing boxes inside gaps of at most max_gap frames."""
    boxes = list(boxes)
    valid = [i for i, b in enumerate(boxes) if b is not None]
    for a, b in zip(valid, valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                boxes[a + k] = [(1 - t) * u + t * v for u, v in zip(boxes[a], boxes[b])]
    return boxes


def contiguous_runs(boxes):
    runs, start = [], None
    for i, b in enumerate(boxes + [None]):
        if b is not None and start is None:
            start = i
        elif b is None and start is not None:
            runs.append((start, i))
            start = None
    return runs


def windows(start, end, length, min_len, max_windows=None):
    out = []
    for s in range(start, end, length):
        e = min(s + length, end)
        if e - s >= min_len:
            out.append((s, e))
    return out[:max_windows] if max_windows else out


def smooth(x, win):
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or len(x) < 3:
        return x
    win = min(win, len(x) if len(x) % 2 else len(x) - 1)
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def _ceil_to(v, m):
    return int(math.ceil(v / m) * m)


def _floor_to(v, m):
    return int(math.floor(v / m) * m)


def plan_window(boxes, frame_hw, *, scale=4, margin=1.25, align=32,
                min_side=128, max_side=512, pad=16, smooth_win=5):
    """Plan a square crop for every frame of a window.

    Returns a dict (JSON-serialisable) or None if the target is too small.
      side      HR tube side after any downscaling (multiple of align)
      side_raw  side of the crop taken from the original frame
      pad/pad_raw  context padding used only during degradation
      x0, y0    per-frame top-left of the crop in original-frame pixels
    """
    assert align % scale == 0 and pad % scale == 0
    H, W = frame_hw
    b = np.asarray(boxes, dtype=np.float64)
    cx = smooth((b[:, 0] + b[:, 2]) / 2, smooth_win)
    cy = smooth((b[:, 1] + b[:, 3]) / 2, smooth_win)
    extent = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    side_raw = int(math.ceil(margin * smooth(extent, smooth_win).max()))
    side_raw = min(side_raw, min(H, W))
    if side_raw < min_side:
        return None

    if side_raw <= max_side:
        up = _ceil_to(side_raw, align)
        side = up if up <= min(H, W) else _floor_to(side_raw, align)
        side_raw, factor = side, 1.0
    else:
        side, factor = max_side, max_side / side_raw
    pad_raw = int(round(pad / factor))

    x0 = np.clip(np.rint(cx - side_raw / 2), 0, W - side_raw).astype(int)
    y0 = np.clip(np.rint(cy - side_raw / 2), 0, H - side_raw).astype(int)
    return dict(side=side, side_raw=side_raw, factor=factor, pad=pad, pad_raw=pad_raw,
                x0=x0.tolist(), y0=y0.tolist(), frame_hw=[H, W], scale=scale)


def crop_reflect(img, x0, y0, w, h):
    """Crop with reflect padding wherever the window leaves the frame."""
    H, W = img.shape[:2]
    l, t = max(0, -x0), max(0, -y0)
    r, btm = max(0, x0 + w - W), max(0, y0 + h - H)
    if l or t or r or btm:
        widths = ((t, btm), (l, r)) + ((0, 0),) * (img.ndim - 2)
        img = np.pad(img, widths, mode="reflect")
        x0, y0 = x0 + l, y0 + t
    return img[y0:y0 + h, x0:x0 + w]


def extract_padded(frame, plan, i):
    """HR window with context padding, at tube resolution: (side+2pad)^2."""
    sr, pr = plan["side_raw"], plan["pad_raw"]
    win = crop_reflect(frame, plan["x0"][i] - pr, plan["y0"][i] - pr, sr + 2 * pr, sr + 2 * pr)
    out = plan["side"] + 2 * plan["pad"]
    if win.shape[0] != out:
        win = cv2.resize(win, (out, out), interpolation=cv2.INTER_AREA)
    return win


def extract_mask(mask, plan, i):
    sr = plan["side_raw"]
    m = mask[plan["y0"][i]:plan["y0"][i] + sr, plan["x0"][i]:plan["x0"][i] + sr].astype(np.uint8)
    if sr != plan["side"]:
        m = cv2.resize(m, (plan["side"], plan["side"]), interpolation=cv2.INTER_NEAREST)
    return m


def unpad(img, pad):
    return img[pad:img.shape[0] - pad, pad:img.shape[1] - pad] if pad else img

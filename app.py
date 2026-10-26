"""Targeted video SR demo (the live test interface for the viva).

  python app.py            # local
  python app.py --share    # public link from Colab

Flow (slide 19 of the prep deck): upload a clip -> click two corners of the
target on frame 0 (or type the box) -> Restore. Shows baseline vs ours side by
side in one synced video, a zoomed view of the target, and region metrics when
ground truth exists.

Two modes:
  "HD clip (has ground truth)" - the upload is treated as HR; we degrade it to
      LR ourselves, restore, and score against the original. Use for metrics.
  "Low-res clip (real world)"  - the upload IS the input; no metrics, only a
      visual comparison. Use for reviewer-supplied test clips.
"""
import argparse
import tempfile
import time

import cv2
import gradio as gr
import numpy as np

from tsr import baselines, targeted, targets
from tsr.io import read_frames

_CACHE = {}
OURS = ["ours_svd", "sd_x4", "realesrgan", "bicubic"]
BASES = ["bicubic", "realesrgan"]
MAX_LR_LONG_SIDE = 480  # real-world mode: bigger inputs are shrunk so ×4 output stays sane


def get_method(name, strength=0.5):
    key = (name, strength if name == "ours_svd" else None)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[0] == "ours_svd"] if name == "ours_svd" else []:
            del _CACHE[k]  # keep only one SVD in memory
        kw = {"strength": strength} if name == "ours_svd" else {}
        _CACHE[key] = baselines.build(name, **kw)
    return _CACHE[key]


def load(video, max_frames):
    frames, _ = read_frames(video, max_frames=int(max_frames))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    if not frames:
        raise gr.Error("Could not read any frames from this video.")
    return frames, fps


def on_upload(video, max_frames):
    if video is None:
        return None, None, [], 0, 0, 0, 0
    frames, _ = load(video, 1)
    h, w = frames[0].shape[:2]
    s = min(w, h) // 4
    return frames[0], frames[0], [], w // 2 - s // 2, h // 2 - s // 2, s, s


def draw_box(img, x, y, w, h):
    if img is None:
        return None
    out = img.copy()
    t = max(2, img.shape[1] // 300)
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 170, 0), t)
    return out


def on_click(frame0, pts, evt: gr.SelectData):
    pts = (pts or []) + [tuple(evt.index)]
    if len(pts) < 2:
        return gr.update(), pts, gr.update(), gr.update(), gr.update(), gr.update()
    (x0, y0), (x1, y1) = pts[-2:]
    x, y, w, h = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
    return draw_box(frame0, x, y, w, h), [], x, y, w, h


def run(video, mode, x, y, w, h, max_frames, ours_name, base_name, tracker, degradation,
        strength, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    if w < 8 or h < 8:
        raise gr.Error("Draw a box first (click two corners on frame 0).")
    t0 = time.time()
    progress(0.05, desc="Reading frames")
    frames, fps = load(video, max_frames)
    hd = mode.startswith("HD")
    scale = 4
    if hd:
        progress(0.1, desc=f"Making LR input ({degradation})")
        lr, gt = targeted.make_lr(frames, degradation)
        box0 = [x / scale, y / scale, (x + w) / scale, (y + h) / scale]
    else:
        lr, gt = frames, None
        f = min(1.0, MAX_LR_LONG_SIDE / max(lr[0].shape[:2]))
        if f < 1:
            lr = [cv2.resize(fr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for fr in lr]
        box0 = [x * f, y * f, (x + w) * f, (y + h) * f]

    progress(0.2, desc=f"Tracking target ({tracker})")
    if tracker == "SAM 2":
        if "sam2" not in _CACHE:
            _CACHE["sam2"] = targets.SAM2Tracker()
        masks = _CACHE["sam2"].track(lr, {1: box0})[1]
    else:
        masks = targets.track_boxes_mil(lr, {1: box0})[1]
    boxes = targets_to_boxes(masks, box0)

    progress(0.35, desc=f"Restoring the target with {ours_name}")
    res = targeted.restore_clip(lr, boxes, get_method(ours_name, strength), get_method(base_name),
                                scale=scale)
    progress(0.85, desc="Writing videos")

    plan = res["plan"]
    tmp = tempfile.mkdtemp()
    side = []
    for i, (b, o) in enumerate(zip(res["base"], res["ours"])):
        n = plan["side_raw"] * scale
        bx, by = plan["x0"][i] * scale, plan["y0"][i] * scale
        b = cv2.rectangle(b.copy(), (bx, by), (bx + n, by + n), (255, 170, 0), 2)
        o = cv2.rectangle(o.copy(), (bx, by), (bx + n, by + n), (255, 170, 0), 2)
        side.append(np.hstack([targeted.label(b, f"Baseline: {base_name} (full frame)"),
                               targeted.label(o, f"Ours: {ours_name} on target")]))
    sbs = targeted.write_video(_fit(side, 1920), f"{tmp}/side_by_side.mp4", fps)

    crops = {"LR input": [cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)
                          for t in res["tube_lr"]],
             "Baseline": targeted.tube_crops(res["base"], plan, scale),
             "Ours": targeted.tube_crops(res["ours"], plan, scale)}
    if gt is not None:
        crops["Ground truth"] = targeted.tube_crops(gt, plan, scale)
    zoom = [np.hstack([targeted.label(cv2.resize(c[i], (256, 256), interpolation=cv2.INTER_AREA
                                                 if c[i].shape[0] > 256 else cv2.INTER_CUBIC), k)
                       for k, c in crops.items()]) for i in range(len(lr))]
    zv = targeted.write_video(zoom, f"{tmp}/zoom.mp4", fps)

    table = None
    if gt is not None:
        progress(0.92, desc="Computing region metrics")
        lp = _lpips()
        m = targeted.compare(res, gt, boxes, lp, scale)
        cols = [("PSNR (box) ↑", "psnr_mask"), ("SSIM (box) ↑", "ssim_mask"),
                ("LPIPS (box) ↓", "lpips_mask"), ("tLPIPS ↓", "tlpips"),
                ("Warp err ×1e3 ↓", "warp"), ("Seam ↓", "seam")]
        table = [[lbl, _fmt(m["base"][k]), _fmt(m["ours"][k])] for lbl, k in cols]
    msg = (f"{len(lr)} frames · LR {lr[0].shape[1]}×{lr[0].shape[0]} · target tube "
           f"{plan['side_raw']}px LR → {plan['side_raw'] * scale}px · {time.time() - t0:.0f}s")
    return sbs, zv, table, msg


def targets_to_boxes(masks, box0):
    from tsr.tubes import fill_gaps, mask_to_box
    boxes = fill_gaps([mask_to_box(m) for m in masks], max_gap=len(masks))
    last = box0
    out = []
    for b in boxes:  # if the tracker loses the target, hold the last box
        last = b if b is not None else last
        out.append(last)
    return out


def _fit(frames, max_w):
    w = frames[0].shape[1]
    if w <= max_w:
        return frames
    f = max_w / w
    return [cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for x in frames]


def _lpips():
    if "lpips" not in _CACHE:
        try:
            from tsr.metrics import LPIPS
            _CACHE["lpips"] = LPIPS()
        except ImportError:
            _CACHE["lpips"] = None
    return _CACHE["lpips"]


def _fmt(v):
    return "–" if v is None or np.isnan(v) else f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"


def build_ui():
    with gr.Blocks(title="Targeted Video SR · Team Pairadox") as demo:
        gr.Markdown("## Targeted super-resolution of content in video clips\n"
                    "IE 643 course project · Team Pairadox. Upload a clip, click two corners "
                    "of the target on frame 0, then press **Restore**.")
        frame0 = gr.State(None)
        pts = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Input clip")
                mode = gr.Radio(["HD clip (has ground truth)", "Low-res clip (real world)"],
                                value="HD clip (has ground truth)", label="Input type")
                with gr.Row():
                    x = gr.Number(label="x", precision=0)
                    y = gr.Number(label="y", precision=0)
                    w = gr.Number(label="w", precision=0)
                    h = gr.Number(label="h", precision=0)
                ours = gr.Dropdown(OURS, value="ours_svd", label="Method on the target")
                base = gr.Dropdown(BASES, value="bicubic", label="Baseline (whole frame)")
                with gr.Accordion("Settings", open=False):
                    frames_n = gr.Slider(8, 64, value=28, step=1, label="Frames (consecutive)")
                    tracker = gr.Radio(["MIL", "SAM 2"], value="MIL", label="Tracker")
                    degr = gr.Radio(["bicubic", "realworld"], value="bicubic",
                                    label="Degradation (HD mode)")
                    strength = gr.Slider(0.2, 0.9, value=0.5, step=0.05,
                                         label="Ours: strength (higher = more generated detail)")
                btn = gr.Button("Restore", variant="primary")
            with gr.Column(scale=2):
                img = gr.Image(label="Frame 0: click two corners of the target",
                               interactive=False)
                sbs = gr.Video(label="Baseline vs ours (synced)")
                zoom = gr.Video(label="Target close-up")
                table = gr.Dataframe(headers=["Metric (target region)", "Baseline", "Ours"],
                                     label="Metrics (HD mode only)")
                status = gr.Markdown()

        video.change(on_upload, [video, frames_n], [img, frame0, pts, x, y, w, h])
        img.select(on_click, [frame0, pts], [img, pts, x, y, w, h])
        for comp in (x, y, w, h):
            comp.submit(draw_box, [frame0, x, y, w, h], img)
        btn.click(run, [video, mode, x, y, w, h, frames_n, ours, base, tracker, degr, strength],
                  [sbs, zoom, table, status])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_ui().queue().launch(share=args.share)

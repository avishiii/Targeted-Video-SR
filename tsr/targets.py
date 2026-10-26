"""Where per-frame target masks come from.

  masks   - annotation PNGs (DAVIS layout: palette PNG, pixel value = object id)
  sam2    - SAM 2 video predictor, prompted with boxes on frame 0
            (boxes from a JSON file, or from a YOLO detector)
  box_track - OpenCV MIL tracker from a frame-0 box, as in Experiment 1.
            Masks are the tracked rectangles. Kept so the final results can be
            compared like-for-like with Exp 1, and as the SAM 2 fallback.
  whole   - the whole (centre-square) frame is the target, e.g. CelebV-HQ,
            whose clips are already face-centred

Every function returns {obj_id: [bool mask (H, W) or None per frame]}.
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ------------------------------------------------------------- annotations
def masks_from_annotations(mask_dir, names, min_pixels=64):
    mask_dir = Path(mask_dir)
    per_frame = []
    for n in names:
        f = mask_dir / f"{n}.png"
        per_frame.append(np.array(Image.open(f)) if f.exists() else None)
    ids = set()
    for m in per_frame:
        if m is not None:
            if m.ndim == 3:  # RGB-coded annotations: collapse colours to ids
                raise ValueError(f"{mask_dir}: expected palette/index PNGs, got RGB")
            ids.update(int(v) for v in np.unique(m) if v not in (0, 255))
    out = {}
    for oid in sorted(ids):
        seq = []
        for m in per_frame:
            mm = None if m is None else (m == oid)
            seq.append(mm if mm is not None and mm.sum() >= min_pixels else None)
        out[oid] = seq
    return out


# ------------------------------------------------------------- whole frame
def whole_frame_target(frames):
    h, w = frames[0].shape[:2]
    m = np.ones((h, w), dtype=bool)
    return {1: [m] * len(frames)}


# ------------------------------------------------------------- boxes
def boxes_from_json(path, clip_id):
    """JSON: {"clip_id": [[x, y, w, h], ...]}  (boxes on frame 0, like the demo)."""
    data = {safe_id(k): v for k, v in json.loads(Path(path).read_text()).items()}
    clip_id = safe_id(clip_id)
    # exact match, else prefix: "6655841" matches "6655841-hd_1920_1080_30fps_3"
    key = clip_id if clip_id in data else next((k for k in data if clip_id.startswith(k)), None)
    if key is None:
        return {}
    return {i + 1: [x, y, x + w, y + h] for i, (x, y, w, h) in enumerate(data[key])}


def safe_id(name):
    """Clip ids from file names like '6655841-hd_1920_1080_30fps (3).mp4'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


def track_boxes_mil(frames, boxes0):
    """{obj: [x0, y0, x1, y1]} on frame 0 -> {obj: [rect mask per frame]}."""
    H, W = frames[0].shape[:2]
    out = {}
    for oid, (x0, y0, x1, y1) in boxes0.items():
        tracker = cv2.TrackerMIL_create()
        tracker.init(cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
                     (int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        seq, last = [], (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        for t, f in enumerate(frames):
            if t > 0:
                ok, box = tracker.update(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                last = tuple(int(v) for v in box) if ok else last
            x, y, w, h = last
            m = np.zeros((H, W), bool)
            m[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = True
            seq.append(m if m.any() else None)
        out[oid] = seq
    return out


_YOLO = None


def detect_boxes(frame, conf=0.5, min_area=0.003, max_area=0.25, max_objects=2,
                 model_name="yolov8n.pt"):
    """Pick a few medium-sized, confidently detected objects on frame 0."""
    global _YOLO
    from ultralytics import YOLO
    if _YOLO is None:
        _YOLO = YOLO(model_name)
    res = _YOLO(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), verbose=False)[0]
    H, W = frame.shape[:2]
    cands = []
    for (x0, y0, x1, y1), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        a = (x1 - x0) * (y1 - y0) / (H * W)
        if c >= conf and min_area <= a <= max_area:
            cands.append((float(c), [float(x0), float(y0), float(x1), float(y1)]))
    cands.sort(key=lambda t: -t[0])
    return {i + 1: box for i, (_, box) in enumerate(cands[:max_objects])}


# ------------------------------------------------------------- SAM 2
class SAM2Tracker:
    """Thin wrapper over the official SAM 2 video predictor.

    pip install "git+https://github.com/facebookresearch/sam2.git"
    """

    def __init__(self, model_id="facebook/sam2.1-hiera-small", device="cuda"):
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self.torch = torch
        self.device = device
        self.pred = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    def track(self, frames, boxes0):
        if not boxes0:
            return {}
        torch = self.torch
        T = len(frames)
        out = {oid: [None] * T for oid in boxes0}
        use_bf16 = self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):  # SAM 2 expects a folder of numbered JPEGs
                cv2.imwrite(f"{tmp}/{i:05d}.jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            # bf16 autocast on Ampere+; plain fp32 on a T4 (no fast bf16 there).
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else _null()
            with torch.inference_mode(), ctx:
                state = self.pred.init_state(video_path=tmp, offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
                for oid, box in boxes0.items():
                    self.pred.add_new_points_or_box(inference_state=state, frame_idx=0,
                                                    obj_id=oid, box=np.asarray(box, np.float32))
                for fidx, obj_ids, logits in self.pred.propagate_in_video(state):
                    for k, oid in enumerate(obj_ids):
                        m = (logits[k, 0] > 0).float().cpu().numpy().astype(bool)
                        out[oid][fidx] = m if m.any() else None
                self.pred.reset_state(state)
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

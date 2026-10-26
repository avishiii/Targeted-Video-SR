"""Frame I/O. All frames are RGB uint8 numpy arrays of shape (H, W, 3)."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def image_files(folder):
    return sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in IMG_EXT)


def read_frames(path, max_frames=None, start=0):
    """Read a clip from a folder of images or a video file.

    Returns (frames, names) where names are frame stems used for output files.
    Frames are always consecutive (no subsampling) so temporal metrics are valid.
    """
    path = Path(path)
    if path.is_dir():
        files = image_files(path)[start:]
        if max_frames:
            files = files[:max_frames]
        return [read_image(f) for f in files], [f.stem for f in files]

    if path.suffix.lower() not in VID_EXT:
        raise ValueError(f"Unsupported input: {path}")
    cap = cv2.VideoCapture(str(path))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, [f"{i:05d}" for i in range(start, start + len(frames))]


def read_dir(folder, grayscale=False):
    files = image_files(folder)
    if grayscale:
        return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    return [read_image(f) for f in files]


def write_frames(folder, frames, names=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        name = names[i] if names else f"{i:05d}"
        write_image(folder / f"{name}.png", frame)


def to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

"""Baselines. Every method maps a list of LR frames (one tube) to SR frames ×4.

Methods are video-level (they get the whole tube) so temporal methods fit the
same interface. Heavy imports happen inside each class, so `bicubic` runs
without torch.
"""
from pathlib import Path

import cv2
import numpy as np

from ..io import to_uint8

CACHE = Path.home() / ".cache" / "tsr"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Bicubic:
    name = "bicubic"

    def __init__(self, scale=4, **_):
        self.scale = scale

    def process(self, frames):
        h, w = frames[0].shape[:2]
        size = (w * self.scale, h * self.scale)
        return [cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in frames]


class RealESRGAN:
    """Real-ESRGAN x4plus, applied frame by frame (no temporal modelling).

    Loaded with `spandrel`, which avoids the broken basicsr/torchvision import.
    """
    name = "realesrgan"
    URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    def __init__(self, weights=None, batch=8, half=None, **_):
        import torch
        from spandrel import ModelLoader
        self.torch = torch
        self.device = _device()
        path = Path(weights) if weights else CACHE / "RealESRGAN_x4plus.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self.URL, str(path))
        self.model = ModelLoader().load_from_file(str(path)).to(self.device).eval()
        self.half = (self.device == "cuda") if half is None else half
        if self.half and self.model.supports_half:
            self.model.half()
        else:
            self.half = False
        self.batch = batch

    def process(self, frames):
        torch = self.torch
        out = []
        for i in range(0, len(frames), self.batch):
            x = torch.from_numpy(np.stack(frames[i:i + self.batch])).permute(0, 3, 1, 2)
            x = x.to(self.device).float().div(255)
            if self.half:
                x = x.half()
            with torch.inference_mode():
                y = self.model(x).float().clamp(0, 1)
            out += list(to_uint8(y.mul(255).permute(0, 2, 3, 1).cpu().numpy()))
        return out


class SDx4:
    """Stable Diffusion ×4 upscaler per frame: the Experiment 1 prior, kept as the
    image-prior ablation. Defaults match Exp 1's *targeted* branch (30 steps,
    noise_level 60, empty prompt). The same seed is used for every frame, which
    reduces (but does not remove) flicker.

    Exp 1 used guidance_scale 9 with an empty prompt. With CFG the conditional and
    unconditional branches then receive identical inputs, so guidance changes
    nothing and only doubles the cost. guidance=1.0 gives the same output faster."""
    name = "sd_x4"

    def __init__(self, steps=30, noise_level=60, guidance=1.0, prompt="", seed=0,
                 batch=2, offload=False,
                 model_id="stabilityai/stable-diffusion-x4-upscaler", **_):
        import torch
        from diffusers import StableDiffusionUpscalePipeline
        self.torch = torch
        self.device = _device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
        if offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()  # as in Exp 1; slower, fits any T4
        else:
            self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.kw = dict(num_inference_steps=steps, noise_level=noise_level,
                       guidance_scale=guidance)  # guidance<=1 disables CFG (empty prompt anyway)
        self.prompt, self.seed, self.batch = prompt, seed, batch

    def process(self, frames):
        from PIL import Image
        torch = self.torch
        out = []
        for i in range(0, len(frames), self.batch):
            chunk = [Image.fromarray(f) for f in frames[i:i + self.batch]]
            gens = [torch.Generator(self.device).manual_seed(self.seed) for _ in chunk]
            res = self.pipe(prompt=[self.prompt] * len(chunk), image=chunk,
                            generator=gens, **self.kw).images
            out += [np.asarray(r) for r in res]
        return out


METHODS = {m.name: m for m in (Bicubic, RealESRGAN, SDx4)}


def build(name, **kw):
    if name == "ours_svd":
        from ..vprior import SVDTube
        return SVDTube(**kw)
    if name not in METHODS:
        raise KeyError(f"Unknown method {name}. Available: {sorted(METHODS) + ['ours_svd']} "
                       "(BasicVSR++/RealBasicVSR: scripts/run_mmagic_baseline.py)")
    return METHODS[name](**kw)

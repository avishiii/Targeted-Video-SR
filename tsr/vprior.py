"""Our method: restore a target tube with a FROZEN video-diffusion prior.

Training-free (option 1 on slide 10 of the prep deck). Nothing is trained; the
low-resolution target steers the frozen Stable Video Diffusion (SVD) model
through three mechanisms:

  1. Tube initialisation (SDEdit-style). The bicubic-upsampled LR tube is
     encoded by the frozen VAE and noised only part-way (`strength`). The
     denoiser then restores detail instead of inventing a new video, so the
     content and motion come from the input.
  2. Frame conditioning. SVD is an image-to-video model: every frame of a
     window is conditioned (CLIP embedding + concatenated VAE latent) on the
     window's first frame. Windows overlap by one frame, and window k>0 is
     conditioned on the ALREADY RESTORED last frame of window k-1, with that
     frame's latent used as its initialisation. Detail is carried forward
     instead of being re-invented every window, which limits flicker on clips
     longer than the model window.
  3. Low-frequency consistency. After decoding, the low frequencies are taken
     from the bicubic-upsampled LR and only the high frequencies from the prior
     (the colour/wavelet fix of StableSR [4]). The prior can add texture but
     cannot shift colour or structure away from what the LR shows.

All three temporal layers of SVD act across the window, which is the
motivation for a video prior instead of a per-frame image prior (sd_x4).
Interface matches tsr/baselines: process(list of LR uint8 frames) -> SR frames.
"""
import cv2
import numpy as np


def lowfreq_fix(sr, ref, sigma):
    """Replace the low frequencies of sr with those of ref (both HxWx3 uint8)."""
    if sigma <= 0:
        return sr
    sr_f, ref_f = sr.astype(np.float32), ref.astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)  # noqa: E731
    return np.clip(blur(ref_f) + (sr_f - blur(sr_f)), 0, 255).astype(np.uint8)


def _work_size(h, w, target, mult=64):
    s = target / max(h, w)
    return (max(mult, int(round(w * s / mult)) * mult), max(mult, int(round(h * s / mult)) * mult))


class SVDTube:
    name = "ours_svd"

    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid",
                 work_size=512, window=14, steps=25, strength=0.5, noise_aug=0.02,
                 motion_bucket=127, fps=7, fix_sigma=3.0, seed=0, decode_chunk=2,
                 offload=False, scale=4, **_):
        import torch
        from diffusers import StableVideoDiffusionPipeline
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kw = {"variant": "fp16"} if dtype == torch.float16 else {}
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **kw)
        if offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        if hasattr(self.pipe.unet, "enable_forward_chunking"):
            self.pipe.unet.enable_forward_chunking()  # lower peak memory, same output
        self.pipe.set_progress_bar_config(disable=True)
        for p in (self.pipe.unet, self.pipe.vae, self.pipe.image_encoder):
            p.requires_grad_(False)  # frozen prior
        self.cfg = dict(work_size=work_size, window=window, steps=steps, strength=strength,
                        noise_aug=noise_aug, motion_bucket=motion_bucket, fps=fps,
                        fix_sigma=fix_sigma, seed=seed, decode_chunk=decode_chunk)
        self.scale = scale

    # ------------------------------------------------------------------ core
    def _to_tensor(self, frames):
        x = self.torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
        return x.to(self.device)

    def _run_window(self, init_frames, cond_frame):
        """init_frames: n uint8 frames at work size. Returns n uint8 frames."""
        from PIL import Image
        torch, pipe, c = self.torch, self.pipe, self.cfg
        dev, n = self.device, len(init_frames)
        H, W = init_frames[0].shape[:2]
        gen = torch.Generator(device="cpu").manual_seed(c["seed"])
        with torch.inference_mode():
            cond_pil = Image.fromarray(cond_frame)
            emb = pipe._encode_image(cond_pil, dev, 1, False)
            cond = pipe.video_processor.preprocess(cond_pil, height=H, width=W).to(dev)
            cond = cond + c["noise_aug"] * torch.randn(cond.shape, generator=gen).to(dev, cond.dtype)

            upcast = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
            if upcast:
                pipe.vae.to(dtype=torch.float32)
            cond_lat = pipe._encode_vae_image(cond, dev, 1, False)
            x = self._to_tensor(init_frames).to(pipe.vae.dtype)
            z0 = torch.cat([pipe.vae.encode(x[i:i + 4]).latent_dist.mode()
                            for i in range(0, n, 4)]) * pipe.vae.config.scaling_factor
            if upcast:
                pipe.vae.to(dtype=torch.float16)

            cond_lat = cond_lat.to(emb.dtype).unsqueeze(1).repeat(1, n, 1, 1, 1)
            z0 = z0.to(emb.dtype).unsqueeze(0)  # (1, n, 4, h, w)
            added = pipe._get_add_time_ids(c["fps"] - 1, c["motion_bucket"], c["noise_aug"],
                                           emb.dtype, 1, 1, False).to(dev)

            sched = pipe.scheduler
            sched.set_timesteps(c["steps"], device=dev)
            start = min(max(int(round(c["steps"] * (1 - c["strength"]))), 0), c["steps"] - 1)
            ts = sched.timesteps[start:]
            if hasattr(sched, "set_begin_index"):
                sched.set_begin_index(start)
            noise = torch.randn(z0.shape, generator=gen).to(dev, z0.dtype)
            latents = sched.add_noise(z0, noise, ts[:1])

            for t in ts:
                inp = torch.cat([sched.scale_model_input(latents, t), cond_lat], dim=2)
                pred = pipe.unet(inp, t, encoder_hidden_states=emb, added_time_ids=added,
                                 return_dict=False)[0]
                latents = sched.step(pred, t, latents).prev_sample

            out = pipe.decode_latents(latents, n, c["decode_chunk"])  # (1, 3, n, H, W)
        out = (out[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5
        return list(out.round().byte().cpu().numpy())

    def process(self, frames):
        c, s = self.cfg, self.scale
        h, w = frames[0].shape[:2]
        up = [cv2.resize(f, (w * s, h * s), interpolation=cv2.INTER_CUBIC) for f in frames]
        ws = _work_size(h * s, w * s, c["work_size"])
        work = [cv2.resize(f, ws, interpolation=cv2.INTER_CUBIC) for f in up]

        T, out, idx = len(frames), [None] * len(frames), 0
        while idx < T:
            first = 0 if idx == 0 else idx - 1  # overlap one restored frame
            ids = list(range(first, min(first + c["window"], T)))
            init = [out[j] if out[j] is not None else work[j] for j in ids]
            res = self._run_window(init, init[0])
            for j, r in zip(ids, res):
                if out[j] is None:
                    out[j] = r
            idx = ids[-1] + 1

        final = []
        for o, ref in zip(out, up):
            o = cv2.resize(o, (w * s, h * s), interpolation=cv2.INTER_AREA)
            final.append(lowfreq_fix(o, ref, c["fix_sigma"]))
        return final

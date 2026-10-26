"""Metrics on the target tube, each reported on the whole crop and inside the mask.

  PSNR / SSIM  Y channel (BT.601), as in the VSR literature (BasicVSR++, REDS4)
  LPIPS        AlexNet, spatial map averaged over the region
  tLPIPS       |LPIPS(sr_t, sr_t-1) - LPIPS(gt_t, gt_t-1)|  (TecoGAN definition)
  warp error   ||sr_t - warp(sr_t-1)||^2 using flow estimated on the GROUND TRUTH,
               over non-occluded mask pixels. Using GT flow means every method is
               judged with the same motion, and flow quality no longer depends on
               the method's own artefacts (the problem in Experiment 1).
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def rgb_to_y(img):
    img = img.astype(np.float64)
    return 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0


def psnr(a, b, region=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[region].mean() if region is not None else d.mean()
    return 100.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a_y, b_y, region=None):
    val, smap = structural_similarity(a_y, b_y, data_range=255, full=True,
                                      gaussian_weights=True, sigma=1.5,
                                      use_sample_covariance=False)
    return float(smap[region].mean()) if region is not None else float(val)


class LPIPS:
    """Lazy LPIPS so the rest works without torch."""

    def __init__(self, net="alex", device=None):
        import lpips
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net, spatial=True, verbose=False).to(self.device).eval()

    def _t(self, img):
        x = self.torch.from_numpy(img).permute(2, 0, 1)[None].float().div(127.5).sub(1)
        return x.to(self.device)

    def map(self, a, b):
        with self.torch.inference_mode():
            return self.model(self._t(a), self._t(b))[0, 0].cpu().numpy()


def _region_mean(m, region):
    return float(m[region].mean()) if region is not None else float(m.mean())


def flow_pair(gt_t, gt_prev):
    """Flow t -> t-1 on GT, plus a forward-backward occlusion mask."""
    g1 = cv2.cvtColor(gt_t, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(gt_prev, cv2.COLOR_RGB2GRAY)
    args = (None, 0.5, 4, 21, 3, 5, 1.1, 0)
    fw = cv2.calcOpticalFlowFarneback(g1, g0, *args)  # g1(p) ~ g0(p + fw(p))
    bw = cv2.calcOpticalFlowFarneback(g0, g1, *args)
    h, w = g1.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + fw[..., 0], gy + fw[..., 1]
    bw_w = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff = ((fw + bw_w) ** 2).sum(-1)
    mag = (fw ** 2).sum(-1) + (bw_w ** 2).sum(-1)
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    valid = (diff < 0.01 * mag + 0.5) & inside
    return (mx, my), valid


def warp_error(cur, prev, maps, valid):
    warped = cv2.remap(prev.astype(np.float32), maps[0], maps[1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    err = ((cur.astype(np.float32) - warped) / 255.0) ** 2
    err = err.mean(-1)
    return float(err[valid].mean()) if valid.any() else np.nan


def evaluate_tube(sr, gt, masks, lpips_model=None, flows=None, min_mask_frac=0.01):
    """sr, gt: lists of HxWx3 uint8. masks: list of HxW uint8 (0/255).

    Returns a dict of means over frames. Mask-level values skip frames whose
    mask covers < min_mask_frac of the crop (target briefly lost/occluded).
    """
    rows = {k: [] for k in ["psnr", "psnr_mask", "ssim", "ssim_mask",
                            "lpips", "lpips_mask", "tlpips", "warp", "warp_gt"]}
    for t, (s, g, m) in enumerate(zip(sr, gt, masks)):
        region = m > 127
        use_mask = region.mean() >= min_mask_frac
        sy, gy = rgb_to_y(s), rgb_to_y(g)
        rows["psnr"].append(psnr(sy, gy))
        rows["ssim"].append(ssim(sy, gy))
        if use_mask:
            rows["psnr_mask"].append(psnr(sy, gy, region))
            rows["ssim_mask"].append(ssim(sy, gy, region))
        if lpips_model is not None:
            lm = lpips_model.map(s, g)
            rows["lpips"].append(_region_mean(lm, None))
            if use_mask:
                rows["lpips_mask"].append(_region_mean(lm, region))
            if t > 0:
                d_sr = lpips_model.map(s, sr[t - 1]).mean()
                d_gt = lpips_model.map(g, gt[t - 1]).mean()
                rows["tlpips"].append(abs(float(d_sr) - float(d_gt)))
        if t > 0 and flows is not None:
            maps, valid = flows[t - 1]
            v = valid & region if use_mask else valid
            rows["warp"].append(warp_error(s, sr[t - 1], maps, v))
            rows["warp_gt"].append(warp_error(g, gt[t - 1], maps, v))
    out = {}
    for k, v in rows.items():
        v = [x for x in v if x is not None and not np.isnan(x)]
        out[k] = float(np.mean(v)) if v else np.nan
    for k in ("warp", "warp_gt"):  # report ×1e3 for readability
        out[k] = out[k] * 1e3 if not np.isnan(out[k]) else np.nan
    return out

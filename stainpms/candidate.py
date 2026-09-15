"""Hematoxylin residual mining and GT assignment for training-only PMS."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_dilation, binary_opening, disk


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _to_uint8_rgb_hwc(image) -> np.ndarray:
    """Coerce image (tensor or numpy, CHW or HWC, normalized or raw) to uint8 HWC RGB."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 4:
        arr = arr[0]

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr[:3], (1, 2, 0))

    arr = arr.astype(np.float32)

    # Heuristic: if values look like ImageNet-normalized (negative or > 1.5),
    # de-normalize. Otherwise if in [0, 1.5], scale to [0, 255]. Otherwise assume
    # already 0..255.
    if arr.min() < -0.01 or arr.max() > 1.5:
        # Looks ImageNet-normalized OR already in [0, 255]. Try de-normalize first
        # and check if result lies in [0, 1].
        denorm = arr * _IMAGENET_STD + _IMAGENET_MEAN
        if -0.05 <= denorm.min() and denorm.max() <= 1.05:
            arr = denorm.clip(0.0, 1.0) * 255.0
        else:
            arr = arr.clip(0.0, 255.0)
    else:
        arr = arr.clip(0.0, 1.0) * 255.0

    return arr.round().astype(np.uint8)[..., :3]


def _robust_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


def compute_h_evidence(image, sigma: float = 1.0) -> np.ndarray:
    """Compute hematoxylin evidence map in [0, 1] from an RGB crop."""
    rgb_uint8 = _to_uint8_rgb_hwc(image)
    rgb01 = rgb_uint8.astype(np.float32) / 255.0
    hed = rgb2hed(rgb01)
    h = _robust_normalize(hed[..., 0])
    if sigma > 0:
        h = gaussian(h, sigma=sigma, preserve_range=True)
    return h.astype(np.float32).clip(0.0, 1.0)


def _instance_map_to_numpy(inst_map) -> np.ndarray:
    if isinstance(inst_map, torch.Tensor):
        m = inst_map.detach().cpu().numpy()
    else:
        m = np.asarray(inst_map)
    if m.ndim == 3:
        # (1, H, W) or (H, W, 1)
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            # (N, H, W) one-hot per-instance stack -> flatten to instance id map
            ids = np.arange(1, m.shape[0] + 1, dtype=np.int32).reshape(-1, 1, 1)
            m = (m.astype(bool) * ids).max(axis=0)
    return m.astype(np.int32)


def compute_b_candidates_oncrop(
    image,
    inst_map,
    *,
    baseline_inst_map=None,
    baseline_dilate_radius: int = 5,
    top_k: int = 20,
    min_distance: int = 12,
    open_disk: int = 2,
    sigma: float = 1.0,
    gt_match_radius: int = 0,
    return_evidence: bool = False,
    return_gt_inst_ids: bool = False,
    keep_negative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mine residual stain peaks, then assign GT instance IDs.

    Coverage is dilated and subtracted before selecting at most ``top_k``
    peaks. GT assignment searches a square neighborhood of half-width
    ``gt_match_radius`` and selects its nearest foreground pixel.
    With ``keep_negative=True``, unmatched peaks remain with instance ID 0.
    Returns xy coordinates and normalized positive weights, optionally
    followed by GT instance IDs and the hematoxylin evidence map.
    """
    evidence = compute_h_evidence(image, sigma=sigma)
    H, W = evidence.shape

    inst = _instance_map_to_numpy(inst_map)
    if inst.shape != (H, W):
        raise ValueError(
            f"inst_map shape {inst.shape} does not match image shape ({H}, {W})"
        )

    def _make_empty_result():
        empty_xy = np.empty((0, 2), dtype=np.float32)
        empty_w = np.empty((0,), dtype=np.float32)
        empty_ids = np.empty((0,), dtype=np.int32)
        result = (empty_xy, empty_w)
        if return_gt_inst_ids:
            result = result + (empty_ids,)
        if return_evidence:
            result = result + (evidence,)
        return result

    # Otsu threshold may fail if image is uniform. Guard with std check.
    if float(evidence.std()) < 1e-6:
        return _make_empty_result()

    otsu_thr = float(threshold_otsu(evidence))
    binary = evidence >= otsu_thr
    binary = binary_fill_holes(binary)
    if open_disk > 0:
        binary = binary_opening(binary, footprint=disk(open_disk))

    # Subtract baseline coverage so PMS focuses on candidate FNs.
    baseline_arr = None
    if baseline_inst_map is not None:
        baseline_arr = _instance_map_to_numpy(baseline_inst_map)
        baseline_bin = baseline_arr > 0
        if baseline_bin.shape != (H, W):
            raise ValueError(
                f"baseline_inst_map shape {baseline_bin.shape} != image ({H}, {W})"
            )
        if baseline_dilate_radius > 0:
            baseline_dil = binary_dilation(baseline_bin, footprint=disk(baseline_dilate_radius))
        else:
            baseline_dil = baseline_bin
        binary = binary & (~baseline_dil)

    masked = evidence * binary
    coords_yx = peak_local_max(
        masked,
        min_distance=min_distance,
        threshold_abs=0,
        exclude_border=False,
    )

    if len(coords_yx) > 0:
        scores = masked[coords_yx[:, 0], coords_yx[:, 1]].astype(np.float32)
        order = np.argsort(-scores)
        if top_k > 0:
            order = order[:top_k]
        coords_yx = coords_yx[order]
        scores = scores[order]
    else:
        coords_yx = np.empty((0, 2), dtype=np.int64)
        scores = np.empty((0,), dtype=np.float32)

    if len(coords_yx) == 0:
        return _make_empty_result()

    # GT filter + lookup associated GT instance id for each surviving candidate.
    # When gt_match_radius == 0: candidate must lie strictly inside a GT pixel,
    #   inst_id = inst[y, x].
    # When gt_match_radius > 0: candidate accepted iff its (2r+1) window contains
    #   any GT pixel. We then pick the NEAREST GT pixel in that window and use
    #   its inst_id (needed downstream for PMS to look up the right GT mask).
    ys = coords_yx[:, 0]
    xs = coords_yx[:, 1]
    keep = np.zeros(len(coords_yx), dtype=bool)
    inst_ids = np.zeros(len(coords_yx), dtype=np.int32)

    if gt_match_radius <= 0:
        strict = inst[ys, xs]
        keep = strict > 0
        inst_ids = strict.astype(np.int32)
    else:
        r = int(gt_match_radius)
        for i, (y, x) in enumerate(zip(ys, xs)):
            y0 = max(0, y - r); y1 = min(H, y + r + 1)
            x0 = max(0, x - r); x1 = min(W, x + r + 1)
            window = inst[y0:y1, x0:x1]
            if (window > 0).any():
                keep[i] = True
                ny, nx = np.where(window > 0)
                dists2 = (ny - (y - y0)) ** 2 + (nx - (x - x0)) ** 2
                nearest = int(np.argmin(dists2))
                inst_ids[i] = int(window[ny[nearest], nx[nearest]])

    if not keep_negative:
        coords_yx = coords_yx[keep]
        scores = scores[keep]
        inst_ids = inst_ids[keep]
    # When keep_negative=True, all top_k peaks are returned; inst_ids[i] == 0
    # marks the peak as a negative (not matched to any GT instance).

    if len(coords_yx) == 0:
        return _make_empty_result()

    coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
    # Weights sum to 1 over the positive subset only. Caller can re-normalize if needed.
    pos_mask = inst_ids > 0
    pos_score_sum = float(scores[pos_mask].sum()) if pos_mask.any() else 0.0
    weights = np.zeros_like(scores)
    if pos_score_sum > 0:
        weights[pos_mask] = scores[pos_mask] / (pos_score_sum + 1e-8)
    weights = weights.astype(np.float32)

    result = (coords_xy, weights)
    if return_gt_inst_ids:
        result = result + (inst_ids,)
    if return_evidence:
        result = result + (evidence,)
    return result


def compute_baseline_center_candidates(
    baseline_inst_map,
    inst_map,
    gt_match_radius: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric centers of baseline-predicted cells as extra positive prompts.

    For each instance in baseline_inst_map, take its centroid; look up the
    matching GT instance in inst_map (within gt_match_radius). Centers that
    do not match any GT are dropped. This produces prompts that are (a) on
    the actual cell center (vs H-peaks which sit on chromatin-bright spots
    off-center), and (b) paired with a high-quality GT mask. Used by PMS
    baseline-preservation: alongside the noisy H-peak prompts, also
    supervise (centered_prompt, GT_mask) so SAM2 decoder mask quality on
    already-detected cells does not degrade during fine-tune.

    Returns
    -------
    coords_xy : (M, 2) float32 xy
    weights   : (M,) float32, uniform = 1/M (sum=1)
    inst_ids  : (M,) int32, matched GT instance ids
    """
    bl = _instance_map_to_numpy(baseline_inst_map)
    inst = _instance_map_to_numpy(inst_map)
    if bl.shape != inst.shape:
        raise ValueError(
            f"baseline shape {bl.shape} != inst shape {inst.shape}"
        )
    H, W = bl.shape

    cell_ids = np.unique(bl)
    cell_ids = cell_ids[cell_ids > 0]
    if len(cell_ids) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    coords_xy_list = []
    inst_ids_list = []
    for cell_id in cell_ids:
        ys, xs = np.where(bl == cell_id)
        if len(ys) == 0:
            continue
        cy = max(0, min(H - 1, int(round(float(ys.mean())))))
        cx = max(0, min(W - 1, int(round(float(xs.mean())))))
        # Match center against GT inst_map (with radius)
        if gt_match_radius <= 0:
            gt_id = int(inst[cy, cx])
            if gt_id == 0:
                continue
        else:
            r = int(gt_match_radius)
            y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
            x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
            window = inst[y0:y1, x0:x1]
            if not (window > 0).any():
                continue
            ny, nx = np.where(window > 0)
            dists2 = (ny - (cy - y0)) ** 2 + (nx - (cx - x0)) ** 2
            nearest = int(np.argmin(dists2))
            gt_id = int(window[ny[nearest], nx[nearest]])
        coords_xy_list.append([cx, cy])
        inst_ids_list.append(gt_id)

    if len(coords_xy_list) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    coords_xy = np.asarray(coords_xy_list, dtype=np.float32)
    inst_ids = np.asarray(inst_ids_list, dtype=np.int32)
    M = len(coords_xy)
    weights = np.full(M, 1.0 / M, dtype=np.float32)
    return coords_xy, weights, inst_ids

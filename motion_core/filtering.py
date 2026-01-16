import numpy as np
from typing import List, Tuple, Optional
import cv2




BASE_PROFILE_W = 1920  # temporary placeholder
BASE_PROFILE_H = 1080  # temporary placeholder
CAMERA_PRESETS = {}    # temporary placeholder



def _water_polygon(preset_data: dict, config: "Config") -> Tuple[Optional[np.ndarray], int]:
    """Build a single polygon in BASE_PROFILE_W/H coordinates, applying shifts + x-scale."""
    profile = preset_data.get("water_profile") or []
    if not profile:
        return (None, 0)

    dx  = int(getattr(config, "WATER_SHIFT_X", 0))
    dyt = int(getattr(config, "WATER_SHIFT_Y_TOP", 0))
    dyb = int(getattr(config, "WATER_SHIFT_Y_BOTTOM", 0))
    scale_x  = float(getattr(config, "WATER_SCALE_X", 1.0))
    anchor_s = str(getattr(config, "WATER_ANCHOR_X", "center")).lower()

    if anchor_s == "left":
        ax = 0.0
    elif anchor_s == "right":
        ax = float(BASE_PROFILE_W)
    else:
        ax = float(BASE_PROFILE_W) / 2.0

    top: List[Tuple[int, int]] = []
    bot: List[Tuple[int, int]] = []
    for (x, y, z) in profile:
        xt = float(x + dx)
        yt = int(y + dyt)
        zb = int(z + dyb)

        xs = ax + (xt - ax) * scale_x
        xs_i = int(min(BASE_PROFILE_W - 1, max(0, round(xs))))
        yt   = int(min(2000, max(-2000, yt)))
        zb   = int(min(2000, max(-2000, zb)))

        top.append((xs_i, yt))
        bot.append((xs_i, zb))

    pts = top + bot[::-1]
    if len(pts) < 3:
        return (None, 0)

    return (np.asarray(pts, dtype=np.int32), len(pts))

# GS 2
def threshold_mag(mag, threshold):
    return (mag > threshold).astype(np.uint8) * 255


def apply_keep_mask(motion_mask, keep_mask):
    if keep_mask is None:
        return motion_mask
    return cv2.bitwise_and(motion_mask, keep_mask)

# ======================================================================================
# MASK BUILD
# ======================================================================================

def build_keep_mask(config: "Config", frame_shape: Tuple[int, int], preset: str) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    preset_data = CAMERA_PRESETS.get(preset, {"water_profile": [], "rois": []})

    # ROI (unscaled; assumes 1920x1080 recordings — keep as-is for now)
    rois = preset_data.get("rois") or []
    if config.ROI_FILTER_ENABLED and rois:
        mode = (config.ROI_FILTER_MODE or "exclude").lower()
        if mode == "include":
            mask[:, :] = 0
            for (x1, y1, x2, y2) in rois:
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
        else:
            for (x1, y1, x2, y2) in rois:
                mask[int(y1):int(y2), int(x1):int(x2)] = 0

    # Water
    poly, npts = (None, 0)
    if config.WATER_MASK_ENABLED and (preset_data.get("water_profile") or []):
        poly, npts = _water_polygon(preset_data, config)

    if poly is not None and npts >= 3:
        sx = w / float(BASE_PROFILE_W)
        sy = h / float(BASE_PROFILE_H)
        poly_scaled = poly.copy()
        poly_scaled[:, 0] = (poly_scaled[:, 0] * sx).astype(np.int32)
        poly_scaled[:, 1] = (poly_scaled[:, 1] * sy).astype(np.int32)

        water_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(water_mask, [poly_scaled], 255)

        k = int(getattr(config, "WATER_MASK_DILATE_PX", 0) or 0)
        if k > 0:
            kernel = np.ones((k, k), np.uint8)
            water_mask = cv2.dilate(water_mask, kernel)

        if (getattr(config, "WATER_MASK_MODE", "exclude") or "exclude") == "exclude":
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(water_mask))
        else:
            mask = cv2.bitwise_and(mask, water_mask)

    return mask

def apply_morphology(mask, kernel_sz=3, open_iters=0, close_iters=0):
    """Apply optional open/close morphology to a binary mask."""
    if mask is None:
        return None

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
    out = mask.copy()

    if open_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iters)

    if close_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iters)

    return out




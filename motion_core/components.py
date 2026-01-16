# components.py


from typing import Tuple, List
import cv2
import numpy as np

from motion_core.filtering import threshold_mag, apply_keep_mask
from motion_core.segmentation import _segments_from_kept_frames


def compute_flow_and_mag(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return flow, mag


def compute_motion_mask(
    mag: np.ndarray,
    threshold: float,
    keep_mask: np.ndarray,
    gating_mode: str
) -> Tuple[np.ndarray, np.ndarray]:

    raw_mask = threshold_mag(mag, threshold)

    if gating_mode == "hard":
        motion_mask = apply_keep_mask(raw_mask, keep_mask)
    else:
        motion_mask = raw_mask

    return raw_mask, motion_mask


def segment_from_kept(kept_frames: List[int], config) -> List[Tuple[int, int]]:
    return _segments_from_kept_frames(kept_frames, config)

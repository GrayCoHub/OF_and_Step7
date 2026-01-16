# flow.py


import cv2
import numpy as np

def compute_farneback_flow(prev_gray, gray):
    """
    Thin wrapper around Farneback optical flow.
    This is a direct copy of the repeated calls in optical_flow_twoPass.py.
    """
    if prev_gray is None or gray is None:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


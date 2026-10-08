"""Task-specific fixed background masks used by the selected video models."""
from pathlib import Path
import cv2
import numpy as np

def mask_external(rgb, kind, roi_path=None):
    image = np.asarray(rgb)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image, 640x480')
    result = image.copy()
    if kind == 'right_edge':
        result[:, 500:] = 96
    elif kind == 'cup_background':
        result[:230, 440:] = 96
        result[:, 500:] = 96
    elif kind == 'fixed_roi':
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if roi is None or roi.shape != (480, 640):
            raise ValueError(f'Invalid 640x480 ROI: {roi_path}')
        result[roi == 0] = 96
    else:
        raise ValueError(f'Unknown mask: {kind}')
    return result

def apply_hand_controls(camera, settings):
    cap = getattr(camera, 'videocapture', None)
    if settings and cap is None:
        raise RuntimeError('Hand VideoCapture is unavailable')
    props = dict(auto_exposure=cv2.CAP_PROP_AUTO_EXPOSURE,
                 exposure=cv2.CAP_PROP_EXPOSURE, contrast=cv2.CAP_PROP_CONTRAST,
                 sharpness=cv2.CAP_PROP_SHARPNESS)
    results = {}
    for name, requested in settings.items():
        success = cap.set(props[name], requested)
        results[name] = dict(requested=requested, actual=cap.get(props[name]),
                             success=bool(success))
    return results

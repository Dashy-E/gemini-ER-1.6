import cv2
import logging

log = logging.getLogger("gemini-er")


def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame


def verify_resolution(cap, expected_w: int, expected_h: int) -> bool:
    """
    Check that the camera actually granted the requested resolution.
    Logs a warning if not — mismatched resolution breaks coordinate transforms.
    Returns True if the resolution matches.
    """
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != expected_w or actual_h != expected_h:
        log.warning(
            f"Camera resolution mismatch: requested {expected_w}×{expected_h}, "
            f"got {actual_w}×{actual_h}. "
            "Coordinate transforms will be incorrect — update FRAME_W/FRAME_H in main.py."
        )
        return False
    log.info(f"Camera resolution verified: {actual_w}×{actual_h}")
    return True

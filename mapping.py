"""
Pixel ↔ world coordinate transforms.

Convention:
  - Origin (0, 0) is at the centre of the camera frame.
  - X increases rightward.
  - Y increases upward (bottom of frame = negative Y).

CALIBRATION NOTE:
  WORLD_SCALE is the most important constant in this file. It is measured in
  metres per pixel and depends entirely on your camera's field of view and its
  physical distance from the workspace surface.

  To calibrate: place two objects on the workspace a known distance apart
  (e.g. 0.20 m), measure their pixel separation in a captured frame, then call:

      scale = calibrate_scale(real_world_distance_m=0.20, pixel_distance_px=<measured>)

  Pass this scale to TrajectoryTracker(world_scale=scale) and pixel_to_world().
  Never rely on the default without verifying it against your actual setup.
"""

# Default: ~50 mm per pixel. Valid only for a specific camera height/FOV.
# MUST be recalibrated for every physical setup change.
WORLD_SCALE: float = 0.05


def calibrate_scale(real_world_distance_m: float, pixel_distance_px: float) -> float:
    """
    Compute the correct world_scale for your camera setup.

    Args:
        real_world_distance_m: Physical distance between two reference points (metres).
        pixel_distance_px:     Pixel distance between those same two points in the frame.

    Returns:
        world_scale in metres/pixel — pass this to pixel_to_world() and TrajectoryTracker.
    """
    if pixel_distance_px <= 0:
        raise ValueError("pixel_distance_px must be > 0")
    return real_world_distance_m / pixel_distance_px


def pixel_to_world(cx: int, cy: int,
                   width: int, height: int,
                   scale: float = WORLD_SCALE) -> tuple[float, float]:
    """
    Convert a pixel centroid to world (robot) coordinates.

    Args:
        cx, cy:  Pixel centroid (OpenCV convention, origin top-left).
        width:   Frame width in pixels.
        height:  Frame height in pixels.
        scale:   Metres per pixel (see WORLD_SCALE / calibrate_scale).

    Returns:
        (world_x, world_y) — frame centre is (0, 0); Y is positive upward.
    """
    world_x = (cx - width / 2) * scale
    world_y = (height / 2 - cy) * scale
    return world_x, world_y


def world_to_pixel(wx: float, wy: float,
                   width: int, height: int,
                   scale: float = WORLD_SCALE) -> tuple[int, int]:
    """
    Inverse of pixel_to_world — project a world position back into pixel space.
    Useful for rendering robot target positions on the live frame.
    """
    cx = int(wx / scale + width / 2)
    cy = int(height / 2 - wy / scale)
    return cx, cy

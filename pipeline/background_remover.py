"""Stage 0: Background removal with automatic detection and manual refinement."""

import numpy as np
from PIL import Image
from rembg import new_session, remove
from scipy import ndimage


def create_rembg_session(model_name: str = "silueta"):
    """Create a rembg session for background removal.

    Use with @st.cache_resource in the app layer to avoid reloading the model.
    """
    return new_session(model_name)


def auto_remove_background(image: Image.Image, session) -> np.ndarray:
    """Detect foreground automatically using rembg.

    Returns a boolean mask (H, W) where True = foreground.
    """
    result = remove(image, session=session)
    # rembg returns RGBA; alpha > 0 means foreground
    alpha = np.array(result.split()[-1])  # alpha channel
    return alpha > 0


def magic_wand_select(
    image: Image.Image, x: int, y: int, tolerance: int = 32, radius: int = 0
) -> np.ndarray:
    """Select a contiguous region of similar color starting from (x, y).

    Like Photoshop's Magic Wand tool. Returns a boolean mask (H, W).
    When radius > 0, the flood fill is limited to pixels within that
    distance (in pixels) from the click point, so clicking on white in
    a corner won't select all white across the entire image.
    """
    rgb = np.array(image.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]

    # Clamp click coordinates
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    seed_color = rgb[y, x]

    # Pixels within color tolerance of the seed
    diff = np.abs(rgb - seed_color)
    within_tol = np.all(diff <= tolerance, axis=2)

    # Restrict to a circular region around the click point
    if radius > 0:
        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        within_tol = within_tol & (dist_sq <= radius * radius)

    # Flood fill: only keep the contiguous region connected to (y, x)
    labeled, n_features = ndimage.label(within_tol)
    seed_label = labeled[y, x]
    if seed_label == 0:
        return np.zeros((h, w), dtype=bool)
    return labeled == seed_label


def create_mask_overlay(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Create a preview image with removed areas dimmed and tinted red.

    Returns an RGB numpy array (H, W, 3) for st.image display.
    """
    rgb = np.array(image.convert("RGB")).copy()

    # Dim + red tint removed areas
    removed = ~mask
    rgb[removed] = (rgb[removed] * 0.3).astype(np.uint8)
    rgb[removed, 0] = np.clip(
        rgb[removed, 0].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)

    return rgb

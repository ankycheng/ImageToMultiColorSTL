"""Stage 1: Color quantization and mask extraction.

Uses KMeans clustering in CIELAB color space for perceptually accurate
color separation, which handles minority colors (like red text on a
yellow/black sign) much better than Pillow's MEDIANCUT.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from .config import PipelineConfig


@dataclass
class ColorLayer:
    """A single color layer with its mask and metadata."""

    color: tuple[int, int, int]  # RGB
    mask: np.ndarray  # boolean 2D array
    hex_color: str
    pixel_count: int
    is_background: bool = False
    components: list[np.ndarray] | None = None  # per-component masks


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) array to CIELAB. Input shape: (..., 3)."""
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # Linearize sRGB
    mask = rgb_norm > 0.04045
    rgb_lin = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # RGB to XYZ (D65)
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = rgb_lin @ mat.T

    # Normalize by D65 white point
    xyz[:, 0] /= 0.95047
    # xyz[:, 1] /= 1.00000  (no-op)
    xyz[:, 2] /= 1.08883

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    lab = np.empty_like(xyz)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0  # L
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])  # a
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])  # b
    return lab


def separate_colors(image: Image.Image, config: PipelineConfig) -> list[ColorLayer]:
    """Quantize image colors via KMeans in LAB space and extract per-color masks.

    Returns a list of ColorLayer sorted by pixel count (largest=background first).
    """
    img = image.convert("RGB")
    w, h = img.size
    pixels_rgb = np.array(img).reshape(-1, 3)  # (N, 3) uint8

    # Convert to LAB for perceptually uniform clustering
    pixels_lab = _rgb_to_lab(pixels_rgb)

    # KMeans clustering in LAB space
    # Use minit='++' for better initialization (k-means++)
    centroids_lab, labels = kmeans2(
        pixels_lab.astype(np.float32),
        config.n_colors,
        minit="++",
        iter=20,
    )

    # Map each cluster back to its mean RGB color
    labels_2d = labels.reshape(h, w)

    layers: list[ColorLayer] = []
    for idx in range(config.n_colors):
        raw_mask = labels_2d == idx
        if not np.any(raw_mask):
            continue

        # Compute mean RGB for this cluster
        cluster_pixels = pixels_rgb[labels == idx]
        mean_rgb = tuple(int(round(v)) for v in cluster_pixels.mean(axis=0))

        # Morphological cleanup to remove anti-aliasing noise
        if config.morph_iterations > 0:
            # Use a larger structuring element (8-connected) for better cleanup
            struct = ndimage.generate_binary_structure(2, 2)
            cleaned = ndimage.binary_closing(
                raw_mask, structure=struct, iterations=config.morph_iterations
            )
            cleaned = ndimage.binary_opening(
                cleaned, structure=struct, iterations=config.morph_iterations
            )
        else:
            cleaned = raw_mask

        pixel_count = int(np.sum(cleaned))
        if pixel_count == 0:
            continue

        r, g, b = mean_rgb
        layers.append(
            ColorLayer(
                color=(r, g, b),
                mask=cleaned,
                hex_color=rgb_to_hex(r, g, b),
                pixel_count=pixel_count,
            )
        )

    # Detect background by border sampling:
    # The color that dominates the image border is the background.
    border_width = max(1, min(h, w) // 50)  # ~2% of shortest side
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_width, :] = True   # top
    border_mask[-border_width:, :] = True  # bottom
    border_mask[:, :border_width] = True   # left
    border_mask[:, -border_width:] = True  # right

    bg_idx = -1
    max_border_ratio = 0.0
    for i, layer in enumerate(layers):
        border_overlap = np.sum(layer.mask & border_mask)
        border_total = np.sum(border_mask)
        ratio = border_overlap / border_total if border_total > 0 else 0
        if ratio > max_border_ratio:
            max_border_ratio = ratio
            bg_idx = i

    # Fallback: if no color covers > 20% of border, use largest area
    if max_border_ratio < 0.2:
        layers.sort(key=lambda l: l.pixel_count, reverse=True)
        bg_idx = 0

    if 0 <= bg_idx < len(layers):
        layers[bg_idx].is_background = True
        # Move background to front
        bg_layer = layers.pop(bg_idx)
        layers.insert(0, bg_layer)

    # Extract connected components for each non-background layer
    for layer in layers:
        if layer.is_background:
            continue
        labeled, n_features = ndimage.label(layer.mask)
        comps = []
        for i in range(1, n_features + 1):
            comp_mask = labeled == i
            if np.sum(comp_mask) < config.potrace_turdsize:
                continue
            comps.append(comp_mask)
        comps.sort(key=lambda m: np.sum(m), reverse=True)
        layer.components = comps

    return layers

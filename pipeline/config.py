from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the PNG-to-multi-color-STL pipeline."""

    # Color quantization
    n_colors: int = 3

    # Physical dimensions
    target_width_mm: float = 100.0
    base_height_mm: float = 1.6
    detail_height_mm: float = 0.4  # default for foreground layers

    # Per-color height overrides: {color_index: height_mm}
    color_heights: dict[int, float] = field(default_factory=dict)

    # Per-component height overrides: {(color_index, component_index): height_mm}
    component_heights: dict[tuple[int, int], float] = field(default_factory=dict)

    # Vectorization
    potrace_turdsize: int = 5  # suppress speckles up to this many pixels
    potrace_alphamax: float = 1.0  # corner smoothness (0=sharp, 1.334=smooth)
    bezier_segments: int = 20  # line segments per bezier curve

    # Morphological cleanup
    morph_iterations: int = 2  # erosion/dilation iterations for anti-alias cleanup

    # Polygon simplification (Shapely)
    simplify_tolerance: float = 0.5  # pixels; 0 = no simplification

    # Flat mode (all layers same height, no protrusion)
    flat_mode: bool = False
    flat_base_color: tuple[int, int, int] | None = None  # None = no base plate
    flat_base_height_mm: float = 0.4

    def get_component_height(self, color_idx: int, comp_idx: int) -> float:
        """Get height for a specific component, with fallback chain."""
        # Per-component override first
        if (color_idx, comp_idx) in self.component_heights:
            return self.component_heights[(color_idx, comp_idx)]
        # Per-color override
        if color_idx in self.color_heights:
            return self.color_heights[color_idx]
        # Global default
        return self.detail_height_mm

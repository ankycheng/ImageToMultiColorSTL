"""Stage 2: Bitmap mask → vector polygons via potrace + Shapely.

Uses the `potrace` (potracer package, import as `potrace`) library.

Potrace tracing behavior on a binary mask:
- Curve 0 is always the full bitmap boundary (background). It has the largest area.
- Subsequent curves alternate between "foreground shape" and "hole inside shape".
- The nesting is: boundary > shape > hole-in-shape > island-in-hole > ...
- We skip the outermost boundary, then use even/odd nesting depth to classify
  remaining curves as exteriors (shapes) or holes.

For our use case (each mask = one color's pixels), we:
1. Skip the bitmap boundary (largest curve).
2. Next-level curves (negative signed area) are our actual shapes (reversed winding).
3. Any deeper curves are holes within those shapes.
"""

from dataclasses import dataclass

import numpy as np
import potrace
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .config import PipelineConfig


@dataclass
class VectorizedLayer:
    """Vectorized polygons for a single color layer."""

    color: tuple[int, int, int]
    hex_color: str
    polygon: MultiPolygon | Polygon | None  # full layer polygon
    component_polygons: list[MultiPolygon | Polygon | None] | None  # per-component
    is_background: bool


def _bezier_point(t: float, p0, p1, p2, p3) -> tuple[float, float]:
    """Evaluate cubic bezier at parameter t."""
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
    )


def _discretize_curve(curve, n_segments: int) -> list[tuple[float, float]]:
    """Convert a potrace Curve to discrete points via its segments."""
    points = []
    start = (curve.start_point.x, curve.start_point.y)
    points.append(start)

    for segment in curve.segments:
        if segment.is_corner:
            c = (segment.c.x, segment.c.y)
            ep = (segment.end_point.x, segment.end_point.y)
            points.append(c)
            points.append(ep)
            start = ep
        else:
            c1 = (segment.c1.x, segment.c1.y)
            c2 = (segment.c2.x, segment.c2.y)
            ep = (segment.end_point.x, segment.end_point.y)
            for i in range(1, n_segments + 1):
                t = i / n_segments
                pt = _bezier_point(t, start, c1, c2, ep)
                points.append(pt)
            start = ep

    return points


def _signed_area(points: list[tuple[float, float]]) -> float:
    """Compute signed area using shoelace formula. Positive = CCW."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _ensure_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ensure points are in CCW order (positive signed area)."""
    if _signed_area(points) < 0:
        return list(reversed(points))
    return points


def _ensure_cw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ensure points are in CW order (negative signed area) for holes."""
    if _signed_area(points) > 0:
        return list(reversed(points))
    return points


def _safe_polygon(coords, holes=None) -> Polygon | None:
    """Create a valid Shapely Polygon, returning None on failure."""
    try:
        poly = Polygon(coords, holes)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            return None
        # make_valid can return GeometryCollection or MultiPolygon
        if isinstance(poly, (GeometryCollection, MultiPolygon)):
            # Extract the largest polygon
            polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
            if not polys:
                return None
            poly = max(polys, key=lambda p: p.area)
        return poly
    except Exception:
        return None


def _trace_mask(
    mask: np.ndarray,
    config: PipelineConfig,
    img_height: int,
    mm_per_pixel: float,
) -> MultiPolygon | Polygon | None:
    """Trace a binary mask into Shapely polygons with hole handling.

    Potrace processes a bitmap where True=foreground. It produces curves where:
    - The first curve (largest positive area) is the bitmap boundary (skip it)
    - The next curves (negative area) are the actual foreground shapes
    - Any further nested curves are holes within those shapes, etc.

    We use signed area to classify: after skipping the boundary, negative-area
    curves become our exteriors (we reverse them to CCW), positive-area curves
    become holes (reversed to CW).
    """
    bmp = potrace.Bitmap(mask.astype(np.bool_))
    path = bmp.trace(
        turdsize=config.potrace_turdsize,
        alphamax=config.potrace_alphamax,
    )

    if not path.curves:
        return None

    # Discretize all curves with their signed areas
    image_area = mask.shape[0] * mask.shape[1]
    curve_data = []  # list of (signed_area, points_px)

    for curve in path.curves:
        pts = _discretize_curve(curve, config.bezier_segments)
        if len(pts) < 3:
            continue
        sa = _signed_area(pts)
        curve_data.append((sa, pts))

    if not curve_data:
        return None

    # Sort by |area| descending to identify the boundary
    curve_data.sort(key=lambda x: abs(x[0]), reverse=True)

    # The largest curve is the bitmap boundary — skip it if it's > 30% of image area
    # (generous threshold to handle partial masks that don't span the full image)
    boundary_area = abs(curve_data[0][0])
    start_idx = 1 if boundary_area > image_area * 0.3 else 0

    if start_idx >= len(curve_data):
        return None

    # Remaining curves: classify as exterior (our shapes) or hole
    # After removing the boundary, negative-area curves are foreground shapes,
    # positive-area curves are holes in those shapes.
    exteriors = []  # points in mm
    holes = []  # points in mm

    for sa, pts in curve_data[start_idx:]:
        # Transform: flip Y and scale to mm
        pts_mm = [
            (x * mm_per_pixel, (img_height - y) * mm_per_pixel) for x, y in pts
        ]

        if sa < 0:
            # Negative area in image coords → this is a foreground shape
            # Reverse to CCW for Shapely exterior
            exteriors.append(_ensure_ccw(pts_mm))
        else:
            # Positive area → hole within a shape
            # Reverse to CW for Shapely hole
            holes.append(_ensure_cw(pts_mm))

    if not exteriors:
        return None

    # Build polygons: match holes to their containing exterior
    # Sort exteriors by area descending (largest first, more likely to contain holes)
    ext_with_area = [(Polygon(e).area, e) for e in exteriors]
    ext_with_area.sort(key=lambda x: x[0], reverse=True)

    polygons = []
    unmatched_holes = list(holes)

    for _, ext_coords in ext_with_area:
        ext_poly = _safe_polygon(ext_coords)
        if ext_poly is None:
            continue

        # Find holes that belong to this exterior
        matched = []
        still_unmatched = []
        for hole_coords in unmatched_holes:
            hole_poly = _safe_polygon(hole_coords)
            if hole_poly and ext_poly.contains(hole_poly.representative_point()):
                matched.append(hole_coords)
            else:
                still_unmatched.append(hole_coords)
        unmatched_holes = still_unmatched

        if matched:
            poly = _safe_polygon(ext_coords, matched)
        else:
            poly = ext_poly

        if poly is not None:
            # Simplify if configured
            if config.simplify_tolerance > 0:
                simplified = poly.simplify(
                    config.simplify_tolerance * mm_per_pixel,
                    preserve_topology=True,
                )
                if not simplified.is_empty:
                    poly = simplified
            polygons.append(poly)

    if not polygons:
        return None

    result = unary_union(polygons)
    if result.is_empty:
        return None
    return result


def vectorize_layers(
    layers: list,  # list[ColorLayer]
    image_size: tuple[int, int],  # (width, height)
    config: PipelineConfig,
) -> list[VectorizedLayer]:
    """Vectorize all color layers into Shapely polygons."""
    img_w, img_h = image_size
    mm_per_pixel = config.target_width_mm / img_w

    results = []
    for layer in layers:
        if layer.is_background:
            # Background is a simple rectangle
            w_mm = img_w * mm_per_pixel
            h_mm = img_h * mm_per_pixel
            rect = Polygon([(0, 0), (w_mm, 0), (w_mm, h_mm), (0, h_mm)])
            results.append(
                VectorizedLayer(
                    color=layer.color,
                    hex_color=layer.hex_color,
                    polygon=rect,
                    component_polygons=None,
                    is_background=True,
                )
            )
            continue

        # Trace full mask
        full_poly = _trace_mask(layer.mask, config, img_h, mm_per_pixel)

        # Trace each connected component
        comp_polys = None
        if layer.components:
            comp_polys = []
            for comp_mask in layer.components:
                cp = _trace_mask(comp_mask, config, img_h, mm_per_pixel)
                comp_polys.append(cp)

        results.append(
            VectorizedLayer(
                color=layer.color,
                hex_color=layer.hex_color,
                polygon=full_poly,
                component_polygons=comp_polys,
                is_background=False,
            )
        )

    return results

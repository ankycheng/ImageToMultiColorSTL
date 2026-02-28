"""Stage 2: Bitmap mask → vector polygons via potrace + Shapely.

Uses potrace for smooth bezier curve tracing, then applies the even-odd
fill rule (via sequential symmetric_difference) to correctly reconstruct
foreground polygons with arbitrarily deep nesting (outlines, holes,
islands-in-holes, etc.).
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


def _ensure_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ensure points are in CCW order (positive signed area)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        return list(reversed(points))
    return points


def _safe_polygon(coords) -> Polygon | None:
    """Create a valid Shapely Polygon, returning None on failure."""
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            return None
        if isinstance(poly, (GeometryCollection, MultiPolygon)):
            polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
            if not polys:
                return None
            poly = max(polys, key=lambda p: p.area)
        return poly
    except Exception:
        return None


def _extract_polygons(geom) -> list[Polygon]:
    """Extract all Polygon instances from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        result = []
        for g in geom.geoms:
            result.extend(_extract_polygons(g))
        return result
    return []


def _trace_mask(
    mask: np.ndarray,
    config: PipelineConfig,
    img_height: int,
    mm_per_pixel: float,
) -> MultiPolygon | Polygon | None:
    """Trace a binary mask into Shapely polygons.

    Potrace traces ALL boundaries between foreground (True) and background
    (False) pixels.  We apply the even-odd fill rule via sequential
    symmetric_difference of every boundary polygon to reconstruct the
    correct foreground geometry — no heuristics needed.
    """
    bmp = potrace.Bitmap(mask.astype(np.bool_))
    path = bmp.trace(
        turdsize=config.potrace_turdsize,
        alphamax=config.potrace_alphamax,
    )

    if not path.curves:
        return None

    # Discretize all curves into polygons (in mm coordinates)
    all_polys = []
    for curve in path.curves:
        pts_px = _discretize_curve(curve, config.bezier_segments)
        if len(pts_px) < 3:
            continue
        pts_mm = [
            (x * mm_per_pixel, (img_height - y) * mm_per_pixel)
            for x, y in pts_px
        ]
        pts_mm = _ensure_ccw(pts_mm)
        poly = _safe_polygon(pts_mm)
        if poly is not None and poly.area > 0.001:
            all_polys.append(poly)

    if not all_polys:
        return None

    h, w = mask.shape
    w_mm = w * mm_per_pixel
    h_mm = h * mm_per_pixel

    # Potrace ALWAYS emits curve 0 as the bitmap boundary (bbox = full
    # image: 0,0 → w,h).  This is NOT part of the actual foreground, so
    # skip it before applying the even-odd fill rule.
    first = all_polys[0]
    bounds = first.bounds  # (minx, miny, maxx, maxy) in mm
    eps = mm_per_pixel * 2  # tolerance
    is_bitmap_boundary = (
        bounds[0] < eps
        and bounds[1] < eps
        and bounds[2] > w_mm - eps
        and bounds[3] > h_mm - eps
    )
    if is_bitmap_boundary:
        all_polys = all_polys[1:]

    if not all_polys:
        return None

    # Apply even-odd fill rule via symmetric_difference.
    # Each boundary curve toggles foreground/background state, so XOR-ing
    # all of them correctly handles any nesting depth:
    #   ring = exterior XOR hole
    #   ring-with-island = exterior XOR hole XOR island
    result = all_polys[0]
    for poly in all_polys[1:]:
        try:
            result = result.symmetric_difference(poly)
        except Exception:
            continue

    # Keep only polygon geometries (drop stray lines/points from XOR)
    polys = _extract_polygons(result)
    if not polys:
        return None

    result = unary_union(polys)

    # Simplify if configured
    if config.simplify_tolerance > 0:
        simplified = result.simplify(
            config.simplify_tolerance * mm_per_pixel,
            preserve_topology=True,
        )
        if not simplified.is_empty:
            result = simplified

    if result.is_empty:
        return None
    return result


def vectorize_layers(
    layers: list,  # list[ColorLayer]
    image_size: tuple[int, int],  # (width, height)
    config: PipelineConfig,
    trace_background: bool = False,
) -> list[VectorizedLayer]:
    """Vectorize all color layers into Shapely polygons.

    Args:
        trace_background: If True, trace the background mask shape (for
            preserving rounded corners etc. when a foreground mask is active).
            If False, use a simple rectangle as the base plate.
    """
    img_w, img_h = image_size
    mm_per_pixel = config.target_width_mm / img_w

    results = []
    for layer in layers:
        if layer.is_background:
            bg_poly = None

            if trace_background:
                bg_poly = _trace_mask(layer.mask, config, img_h, mm_per_pixel)

            if bg_poly is None:
                # Full rectangle as base plate
                w_mm = img_w * mm_per_pixel
                h_mm = img_h * mm_per_pixel
                bg_poly = Polygon([(0, 0), (w_mm, 0), (w_mm, h_mm), (0, h_mm)])

            results.append(
                VectorizedLayer(
                    color=layer.color,
                    hex_color=layer.hex_color,
                    polygon=bg_poly,
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

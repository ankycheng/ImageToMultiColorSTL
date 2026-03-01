"""Stage 3: Extrude 2D polygons into 3D meshes using trimesh.

By default, all components of the same color are merged into a single mesh
(one mesh per color). When per_component=True in config, each connected
component gets its own mesh for individual height control.
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon, box as shapely_box

from .config import PipelineConfig


@dataclass
class MeshLayer:
    """A 3D mesh for a single color (or component)."""

    mesh: trimesh.Trimesh
    color: tuple[int, int, int]
    hex_color: str
    name: str
    is_background: bool
    height_mm: float


def _extrude_polygon(
    polygon: Polygon | MultiPolygon,
    height: float,
    z_offset: float = 0.0,
) -> trimesh.Trimesh | None:
    """Extrude a Shapely polygon to a 3D mesh."""
    if polygon is None or polygon.is_empty:
        return None

    meshes = []

    if isinstance(polygon, MultiPolygon):
        geoms = list(polygon.geoms)
    else:
        geoms = [polygon]

    for geom in geoms:
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue
        if geom.area < 0.01:  # skip tiny polygons (< 0.01 mm²)
            continue
        try:
            mesh = trimesh.creation.extrude_polygon(geom, height)
            if mesh is not None and len(mesh.faces) > 0:
                meshes.append(mesh)
        except (ValueError, RuntimeError, IndexError, TypeError):
            continue

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes)

    if z_offset != 0:
        combined.apply_translation([0, 0, z_offset])

    return combined


def _set_face_color(mesh: trimesh.Trimesh, color: tuple[int, int, int]) -> None:
    """Set uniform face color on a mesh."""
    face_color = list(color) + [255]
    mesh.visual.face_colors = np.tile(
        face_color, (len(mesh.faces), 1)
    ).astype(np.uint8)


def _bounding_box_polygon(vector_layers: list) -> Polygon | None:
    """Compute bounding box polygon from all vector layers (in mm coords).

    Returns None if no valid polygons exist.
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for vl in vector_layers:
        if vl.polygon is None or vl.polygon.is_empty:
            continue
        bx0, by0, bx1, by1 = vl.polygon.bounds
        min_x, min_y = min(min_x, bx0), min(min_y, by0)
        max_x, max_y = max(max_x, bx1), max(max_y, by1)
    if min_x == float("inf"):
        return None
    return shapely_box(min_x, min_y, max_x, max_y)


def extrude_layers(
    vector_layers: list,  # list[VectorizedLayer]
    config: PipelineConfig,
) -> list[MeshLayer]:
    """Extrude all vectorized layers into 3D meshes.

    Flat mode: all layers at uniform height, optional base plate underneath.
    Relief mode (default): background base + raised foreground layers.
    """
    if config.flat_mode:
        return _extrude_flat(vector_layers, config)
    return _extrude_relief(vector_layers, config)


def _extrude_flat(
    vector_layers: list,
    config: PipelineConfig,
) -> list[MeshLayer]:
    """Flat mode: all color tiles at same height, optional base plate."""
    mesh_layers = []
    tile_height = config.base_height_mm
    z_offset = config.flat_base_height_mm if config.flat_base_color else 0.0

    # Optional base plate
    if config.flat_base_color is not None:
        base_poly = _bounding_box_polygon(vector_layers)
        if base_poly is None:
            return mesh_layers
        base_mesh = _extrude_polygon(base_poly, config.flat_base_height_mm)
        if base_mesh is not None:
            _set_face_color(base_mesh, config.flat_base_color)
            hex_color = "#{:02X}{:02X}{:02X}".format(*config.flat_base_color)
            mesh_layers.append(
                MeshLayer(
                    mesh=base_mesh,
                    color=config.flat_base_color,
                    hex_color=hex_color,
                    name=f"{hex_color}_base",
                    is_background=True,
                    height_mm=config.flat_base_height_mm,
                )
            )

    # All color layers at uniform height
    for vlayer in vector_layers:
        if vlayer.polygon is None or vlayer.polygon.is_empty:
            continue
        mesh = _extrude_polygon(vlayer.polygon, tile_height, z_offset=z_offset)
        if mesh is not None:
            _set_face_color(mesh, vlayer.color)
            mesh_layers.append(
                MeshLayer(
                    mesh=mesh,
                    color=vlayer.color,
                    hex_color=vlayer.hex_color,
                    name=f"{vlayer.hex_color}_tile",
                    is_background=False,
                    height_mm=tile_height,
                )
            )

    return mesh_layers


def _extrude_relief(
    vector_layers: list,
    config: PipelineConfig,
) -> list[MeshLayer]:
    """Relief mode (original): base plate + raised foreground layers."""
    mesh_layers = []
    per_component = bool(config.component_heights)

    for color_idx, vlayer in enumerate(vector_layers):
        if vlayer.is_background:
            mesh = _extrude_polygon(vlayer.polygon, config.base_height_mm)
            if mesh is not None:
                _set_face_color(mesh, vlayer.color)
                mesh_layers.append(
                    MeshLayer(
                        mesh=mesh,
                        color=vlayer.color,
                        hex_color=vlayer.hex_color,
                        name=f"{vlayer.hex_color}_base",
                        is_background=True,
                        height_mm=config.base_height_mm,
                    )
                )
            continue

        if per_component and vlayer.component_polygons:
            for comp_idx, comp_poly in enumerate(vlayer.component_polygons):
                if comp_poly is None:
                    continue
                height = config.get_component_height(color_idx, comp_idx)
                mesh = _extrude_polygon(
                    comp_poly, height, z_offset=config.base_height_mm
                )
                if mesh is not None:
                    _set_face_color(mesh, vlayer.color)
                    mesh_layers.append(
                        MeshLayer(
                            mesh=mesh,
                            color=vlayer.color,
                            hex_color=vlayer.hex_color,
                            name=f"{vlayer.hex_color}_part{comp_idx}",
                            is_background=False,
                            height_mm=height,
                        )
                    )
        elif vlayer.polygon is not None:
            height = config.get_component_height(color_idx, 0)
            mesh = _extrude_polygon(
                vlayer.polygon, height, z_offset=config.base_height_mm
            )
            if mesh is not None:
                _set_face_color(mesh, vlayer.color)
                mesh_layers.append(
                    MeshLayer(
                        mesh=mesh,
                        color=vlayer.color,
                        hex_color=vlayer.hex_color,
                        name=f"{vlayer.hex_color}_layer",
                        is_background=False,
                        height_mm=height,
                    )
                )

    return mesh_layers

"""Stage 3: Extrude 2D polygons into 3D meshes using trimesh.

By default, all components of the same color are merged into a single mesh
(one mesh per color). When per_component=True in config, each connected
component gets its own mesh for individual height control.
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon

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
        except Exception:
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


def extrude_layers(
    vector_layers: list,  # list[VectorizedLayer]
    config: PipelineConfig,
) -> list[MeshLayer]:
    """Extrude all vectorized layers into 3D meshes.

    Default: one mesh per color (all components merged).
    If per_component mode: one mesh per connected component.
    """
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
            # Per-component mode: individual meshes
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
            # Merged mode: one mesh per color
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

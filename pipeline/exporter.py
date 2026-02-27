"""Stage 4: Export meshes to STL and 3MF files.

The 3MF export creates a spec-compliant multi-color file that Bambu Studio
(and PrusaSlicer) can read:

Structure:
  <basematerials id="1">        ← color palette
    <base name="Yellow" displaycolor="#FED226" />
    <base name="Black"  displaycolor="#0F0F0A" />
    <base name="Red"    displaycolor="#FF0200" />
  </basematerials>

  <object id="2" pid="1" pindex="0">  ← mesh body for color 0
    <mesh>...</mesh>
  </object>
  <object id="3" pid="1" pindex="1">  ← mesh body for color 1
    <mesh>...</mesh>
  </object>
  ...

  <object id="100" type="model">       ← parent assembly object
    <components>
      <component objectid="2" />
      <component objectid="3" />
      ...
    </components>
  </object>

  <build>
    <item objectid="100" />            ← single build item
  </build>

This results in a single object in the slicer with multiple parts,
each assignable to a different filament/color.
"""

import io
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import trimesh


def _build_3mf_model_xml(mesh_layers: list) -> bytes:
    """Build the 3D/3dmodel.model XML for a multi-color 3MF."""
    NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

    model = Element("model")
    model.set("xmlns", NS)
    model.set("unit", "millimeter")
    model.set("xml:lang", "en-US")

    # Metadata
    meta = SubElement(model, "metadata")
    meta.set("name", "Title")
    meta.text = "MultiColor Model"

    resources = SubElement(model, "resources")

    # --- basematerials: one entry per unique color ---
    # Deduplicate colors while preserving order
    seen_colors = {}  # hex -> index
    color_list = []  # [(hex, name)]
    for ml in mesh_layers:
        if ml.hex_color not in seen_colors:
            seen_colors[ml.hex_color] = len(color_list)
            color_list.append((ml.hex_color, ml.name))

    basemats = SubElement(resources, "basematerials")
    basemats.set("id", "1")
    for hex_color, name in color_list:
        base = SubElement(basemats, "base")
        base.set("name", name)
        base.set("displaycolor", hex_color.upper())

    # --- One <object> per mesh, referencing its material ---
    child_ids = []
    next_id = 2

    for ml in mesh_layers:
        obj = SubElement(resources, "object")
        obj.set("id", str(next_id))
        obj.set("name", ml.name)
        obj.set("type", "model")
        obj.set("pid", "1")  # basematerials id
        obj.set("pindex", str(seen_colors[ml.hex_color]))

        mesh_el = SubElement(obj, "mesh")

        # Vertices
        verts_el = SubElement(mesh_el, "vertices")
        for v in ml.mesh.vertices:
            vert = SubElement(verts_el, "vertex")
            vert.set("x", f"{v[0]:.6f}")
            vert.set("y", f"{v[1]:.6f}")
            vert.set("z", f"{v[2]:.6f}")

        # Triangles
        tris_el = SubElement(mesh_el, "triangles")
        for f in ml.mesh.faces:
            tri = SubElement(tris_el, "triangle")
            tri.set("v1", str(f[0]))
            tri.set("v2", str(f[1]))
            tri.set("v3", str(f[2]))

        child_ids.append(next_id)
        next_id += 1

    # --- Parent assembly object with <components> ---
    parent_id = next_id
    parent_obj = SubElement(resources, "object")
    parent_obj.set("id", str(parent_id))
    parent_obj.set("name", "MultiColorSign")
    parent_obj.set("type", "model")

    components = SubElement(parent_obj, "components")
    for cid in child_ids:
        comp = SubElement(components, "component")
        comp.set("objectid", str(cid))

    # --- Build: single item pointing to parent ---
    build = SubElement(model, "build")
    item = SubElement(build, "item")
    item.set("objectid", str(parent_id))

    xml_bytes = tostring(model, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes).encode("utf-8")


def _build_rels() -> bytes:
    """Build the _rels/.rels file."""
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>"""


def _build_content_types() -> bytes:
    """Build the [Content_Types].xml file."""
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
</Types>"""


def export_3mf(mesh_layers: list, output_path: Path) -> Path:
    """Export all mesh layers as a proper multi-color 3MF file.

    Creates a 3MF with basematerials + component assembly structure that
    Bambu Studio and PrusaSlicer can read for filament assignment.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", _build_3mf_model_xml(mesh_layers))
        zf.writestr("_rels/.rels", _build_rels())
        zf.writestr("[Content_Types].xml", _build_content_types())

    return output_path


def export_stl(mesh_layer, output_dir: Path) -> Path:
    """Export a single MeshLayer as an STL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mesh_layer.name}.stl"
    filepath = output_dir / filename
    mesh_layer.mesh.export(str(filepath), file_type="stl")
    return filepath


def export_all_stl(mesh_layers: list, output_dir: Path) -> list[Path]:
    """Export all mesh layers as individual STL files."""
    paths = []
    for ml in mesh_layers:
        p = export_stl(ml, output_dir)
        paths.append(p)
    return paths


def create_zip(mesh_layers: list, output_dir: Path) -> bytes:
    """Create a ZIP archive containing all STL files + 3MF."""
    stl_paths = export_all_stl(mesh_layers, output_dir)
    mf_path = export_3mf(mesh_layers, output_dir / "combined.3mf")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in stl_paths:
            zf.write(p, p.name)
        zf.write(mf_path, mf_path.name)
    return buf.getvalue()


def validate_meshes(mesh_layers: list) -> list[dict]:
    """Validate all meshes and return a report."""
    reports = []
    for ml in mesh_layers:
        m = ml.mesh
        reports.append(
            {
                "name": ml.name,
                "color": ml.hex_color,
                "faces": len(m.faces),
                "vertices": len(m.vertices),
                "is_watertight": m.is_watertight,
                "is_volume": m.is_volume,
                "volume_mm3": float(m.volume) if m.is_volume else None,
            }
        )
    return reports

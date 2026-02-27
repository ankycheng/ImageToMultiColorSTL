"""Streamlit app: PNG → Multi-Color 3D Print STL Pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from pipeline.color_separator import ColorLayer, separate_colors
from pipeline.config import PipelineConfig


def _checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    """Generate a checkerboard pattern (H, W, 3) as transparency background."""
    rows = np.arange(h) // cell
    cols = np.arange(w) // cell
    grid = (rows[:, None] + cols[None, :]) % 2  # 0 or 1
    light, dark = 180, 120
    img = np.where(grid[..., None], light, dark).astype(np.uint8)
    return np.repeat(img, 3, axis=-1).reshape(h, w, 3)


def _mask_preview(mask: np.ndarray, color: tuple[int, int, int], cell: int = 16) -> np.ndarray:
    """Render mask on a checkerboard background for clear visibility."""
    h, w = mask.shape
    img = _checkerboard(h, w, cell)
    img[mask] = color
    return img
from pipeline.extruder import extrude_layers
from pipeline.exporter import create_zip, export_3mf, export_all_stl, validate_meshes
from pipeline.vectorizer import vectorize_layers

st.set_page_config(page_title="Image → Multi-Color STL", layout="wide")
st.title("PNG → Multi-Color 3D Print Pipeline")
st.caption("Upload a PNG image → auto color separation → vectorize → extrude → download STL/3MF")

# --- Sidebar: Global Parameters ---
st.sidebar.header("Parameters")
n_colors = st.sidebar.slider("Number of colors", 2, 8, 3)
target_width = st.sidebar.number_input("Target width (mm)", 10.0, 500.0, 100.0, step=5.0)
base_height = st.sidebar.number_input("Base plate height (mm)", 0.4, 10.0, 1.6, step=0.2)
default_detail = st.sidebar.number_input("Default detail height (mm)", 0.2, 5.0, 0.4, step=0.1)

st.sidebar.subheader("Advanced")
morph_iter = st.sidebar.slider("Morphological cleanup iterations", 0, 5, 2)
turdsize = st.sidebar.slider("Speckle suppression (px)", 0, 50, 5)
simplify = st.sidebar.slider("Polygon simplification (px)", 0.0, 5.0, 0.5, step=0.1)

# --- File Upload ---
uploaded = st.file_uploader("Upload PNG image", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded is not None:
    image = Image.open(uploaded)
    img_w, img_h = image.size

    # --- Step 1: Show original image ---
    col_orig, col_info = st.columns([2, 1])
    with col_orig:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col_info:
        st.subheader("Image Info")
        st.write(f"Size: {img_w} × {img_h} px")
        mm_per_px = target_width / img_w
        st.write(f"Scale: {mm_per_px:.3f} mm/px")
        st.write(f"Output: {target_width:.1f} × {img_h * mm_per_px:.1f} mm")

    # --- Step 2: Color Separation ---
    st.subheader("Step 1: Color Separation")

    config = PipelineConfig(
        n_colors=n_colors,
        target_width_mm=target_width,
        base_height_mm=base_height,
        detail_height_mm=default_detail,
        morph_iterations=morph_iter,
        potrace_turdsize=turdsize,
        simplify_tolerance=simplify,
    )

    with st.spinner("Separating colors..."):
        layers = separate_colors(image, config)

    # Display color masks
    mask_cols = st.columns(len(layers))
    for i, (col, layer) in enumerate(zip(mask_cols, layers)):
        with col:
            label = "Background" if layer.is_background else f"Color {i}"
            st.markdown(
                f"**{label}** "
                f'<span style="background:{layer.hex_color};padding:2px 12px;border:1px solid #ccc;">&nbsp;</span> '
                f"`{layer.hex_color}`",
                unsafe_allow_html=True,
            )
            # Show mask on checkerboard background
            mask_img = _mask_preview(layer.mask, layer.color)
            st.image(mask_img, use_container_width=True)
            st.caption(f"{layer.pixel_count:,} px")
            if layer.components:
                st.caption(f"{len(layer.components)} components")

    # --- Per-color height settings ---
    st.subheader("Step 2: Height Settings")

    color_heights: dict[int, float] = {}
    component_heights: dict[tuple[int, int], float] = {}

    height_cols = st.columns(len(layers))
    for i, (col, layer) in enumerate(zip(height_cols, layers)):
        with col:
            if layer.is_background:
                st.write(f"**Base plate**: {base_height} mm")
            else:
                h = st.number_input(
                    f"Height for {layer.hex_color}",
                    0.1, 5.0, default_detail, step=0.1,
                    key=f"height_{i}",
                )
                color_heights[i] = h

    # Advanced: per-component overrides
    show_components = st.checkbox("Show per-component height overrides")
    if show_components:
        for i, layer in enumerate(layers):
            if layer.is_background or not layer.components:
                continue
            st.markdown(f"**Components for {layer.hex_color}:**")
            comp_cols = st.columns(min(len(layer.components), 4))
            for j, comp_mask in enumerate(layer.components):
                col_idx = j % len(comp_cols)
                with comp_cols[col_idx]:
                    # Show component mask on checkerboard
                    comp_img = _mask_preview(comp_mask, layer.color)
                    st.image(comp_img, use_container_width=True, caption=f"Part {j}")
                    ch = st.number_input(
                        f"Height (mm)",
                        0.1, 5.0,
                        color_heights.get(i, default_detail),
                        step=0.1,
                        key=f"comp_{i}_{j}",
                    )
                    component_heights[(i, j)] = ch

    # Update config with height settings
    config.color_heights = color_heights
    config.component_heights = component_heights

    # --- Step 3: Generate 3D ---
    st.subheader("Step 3: Generate 3D Model")

    if st.button("Generate STL / 3MF", type="primary"):
        # Vectorize
        with st.spinner("Vectorizing (potrace)..."):
            vector_layers = vectorize_layers(layers, image.size, config)

        # Extrude
        with st.spinner("Extruding 3D meshes..."):
            mesh_layers = extrude_layers(vector_layers, config)

        if not mesh_layers:
            st.error("No meshes generated. Try adjusting parameters.")
            st.stop()

        # Validate
        st.subheader("Mesh Validation")
        reports = validate_meshes(mesh_layers)
        for r in reports:
            icon = "✅" if r["is_watertight"] else "⚠️"
            vol = f", {r['volume_mm3']:.1f} mm³" if r["volume_mm3"] else ""
            st.write(
                f"{icon} **{r['name']}** — "
                f"{r['faces']} faces, {r['vertices']} vertices{vol}"
            )

        # 3D Preview
        st.subheader("3D Preview")
        try:
            from streamlit_stl import stl_from_file

            import trimesh
            combined = trimesh.util.concatenate([ml.mesh for ml in mesh_layers])
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                combined.export(tmp.name, file_type="stl")
                stl_from_file(
                    file_path=tmp.name,
                    color="#808080",
                    auto_rotate=True,
                    height=500,
                )
        except Exception as e:
            st.warning(f"3D preview unavailable: {e}")

        # Export
        st.subheader("Download")
        output_dir = Path(tempfile.mkdtemp())

        # Individual STLs
        stl_paths = export_all_stl(mesh_layers, output_dir)
        for ml, path in zip(mesh_layers, stl_paths):
            stl_data = path.read_bytes()
            st.download_button(
                f"Download {ml.name}.stl ({len(stl_data) // 1024} KB)",
                stl_data,
                file_name=f"{ml.name}.stl",
                mime="application/octet-stream",
                key=f"dl_{ml.name}",
            )

        # 3MF
        mf_path = export_3mf(mesh_layers, output_dir / "combined.3mf")
        st.download_button(
            f"Download combined.3mf ({mf_path.stat().st_size // 1024} KB)",
            mf_path.read_bytes(),
            file_name="combined.3mf",
            mime="application/octet-stream",
            key="dl_3mf",
        )

        # ZIP
        zip_data = create_zip(mesh_layers, output_dir)
        st.download_button(
            f"Download all (ZIP, {len(zip_data) // 1024} KB)",
            zip_data,
            file_name="multicolor_stl.zip",
            mime="application/zip",
            key="dl_zip",
        )

        st.success(f"Generated {len(mesh_layers)} meshes successfully!")

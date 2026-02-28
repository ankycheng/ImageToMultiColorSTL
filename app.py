"""Streamlit app: PNG → Multi-Color 3D Print STL Pipeline."""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from pipeline.background_remover import (
    auto_remove_background,
    create_mask_overlay,
    create_rembg_session,
    magic_wand_select,
)
from pipeline.color_separator import ColorLayer, separate_colors
from pipeline.config import PipelineConfig

# Custom clickable image component with live radius cursor
_COMPONENT_DIR = Path(__file__).parent / "components" / "clickable_image"
_clickable_image = components.declare_component("clickable_image", path=str(_COMPONENT_DIR))


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

    # --- Show original image ---
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # --- Step 0: Background Removal ---
    st.subheader("Step 0: Background Removal")
    enable_bg_removal = st.checkbox("Enable background removal", value=False, key="enable_bg_removal")

    foreground_mask = None

    # Detect alpha channel for shape preservation (e.g., rounded corners)
    alpha_mask = None
    if image.mode in ('RGBA', 'LA', 'PA'):
        alpha_arr = np.array(image.split()[-1])
        if not np.all(alpha_arr > 128):
            alpha_mask = alpha_arr > 128

    if enable_bg_removal:
        @st.cache_resource
        def _get_rembg_session():
            return create_rembg_session("silueta")

        @st.cache_data
        def _auto_remove_bg(image_bytes: bytes, width: int, height: int, contrast: float):
            from PIL import ImageEnhance
            img = Image.open(io.BytesIO(image_bytes))
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            session = _get_rembg_session()
            return auto_remove_background(img, session)

        image_bytes = uploaded.getvalue()

        contrast = st.slider(
            "Pre-detection contrast boost", 0.5, 3.0, 1.0, step=0.1,
            key="contrast_boost",
            help="Increase contrast before auto-detection for better results with low-contrast images",
        )

        with st.spinner("Auto-detecting foreground..."):
            auto_mask = _auto_remove_bg(image_bytes, img_w, img_h, contrast)

        # Initialize mask + undo history in session state
        if "fg_mask" not in st.session_state:
            st.session_state["fg_mask"] = auto_mask
        if "fg_mask_history" not in st.session_state:
            st.session_state["fg_mask_history"] = []

        # Toolbar row 1: mode + buttons
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([3, 1, 1, 1])
        with tool_col1:
            wand_mode = st.radio(
                "Click mode",
                ["Remove area", "Keep area"],
                horizontal=True,
                key="wand_mode",
            )
        with tool_col2:
            undo_disabled = len(st.session_state["fg_mask_history"]) == 0
            if st.button("Undo", key="undo_mask", disabled=undo_disabled):
                st.session_state["fg_mask"] = st.session_state["fg_mask_history"].pop()
        with tool_col3:
            if st.button("Reset", key="reset_mask"):
                st.session_state["fg_mask"] = auto_mask
                st.session_state["fg_mask_history"] = []
        with tool_col4:
            history_len = len(st.session_state["fg_mask_history"])
            if history_len > 0:
                st.caption(f"{history_len} step(s)")

        # Toolbar row 2: tolerance + radius
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            tolerance = st.slider(
                "Tolerance (color similarity)", 1, 100, 32, key="wand_tolerance"
            )
        with param_col2:
            max_radius = max(img_w, img_h)
            radius = st.slider(
                "Search radius (px, 0 = unlimited)",
                0, max_radius, min(200, max_radius // 2), key="wand_radius",
            )

        # Build overlay preview from current mask, with padding for edge access
        current_mask = st.session_state["fg_mask"]
        overlay_arr = create_mask_overlay(image, current_mask)

        # Add 100px dark padding around the overlay so the magic wand
        # radius circle can reach image edges comfortably
        pad = 100
        padded = np.full(
            (img_h + pad * 2, img_w + pad * 2, 3), 30, dtype=np.uint8
        )
        padded[pad:pad + img_h, pad:pad + img_w] = overlay_arr

        overlay_pil = Image.fromarray(padded)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        padded_w = img_w + pad * 2
        padded_h = img_h + pad * 2

        st.caption("Click on the image to select a region (Magic Wand)")

        # Custom clickable image with live radius cursor
        coords = _clickable_image(
            image_b64=image_b64,
            img_w=padded_w,
            img_h=padded_h,
            radius=radius,
            max_display_width=900,
            key="mask_click",
        )

        # Handle click — deduplicate by comparing with last processed click
        if coords is not None:
            last = st.session_state.get("_last_click")
            if last != coords:
                st.session_state["_last_click"] = coords

                # Subtract padding offset to get original image coordinates
                click_x = int(coords["x"]) - pad
                click_y = int(coords["y"]) - pad

                # Use contrast-enhanced image for magic wand detection
                if contrast != 1.0:
                    from PIL import ImageEnhance
                    detect_img = ImageEnhance.Contrast(image).enhance(contrast)
                else:
                    detect_img = image
                wand_region = magic_wand_select(
                    detect_img, click_x, click_y, tolerance, radius=radius
                )

                # Save current mask to history before modifying (undo support)
                st.session_state["fg_mask_history"].append(
                    st.session_state["fg_mask"].copy()
                )
                # Cap history at 30 steps to avoid memory bloat
                if len(st.session_state["fg_mask_history"]) > 30:
                    st.session_state["fg_mask_history"] = st.session_state[
                        "fg_mask_history"
                    ][-30:]

                mask = st.session_state["fg_mask"].copy()
                if wand_mode == "Remove area":
                    mask[wand_region] = False
                else:
                    mask[wand_region] = True
                st.session_state["fg_mask"] = mask
                st.rerun()

        foreground_mask = st.session_state["fg_mask"]

        fg_pct = np.sum(foreground_mask) / foreground_mask.size * 100
        st.caption(f"Foreground: {np.sum(foreground_mask):,} px ({fg_pct:.1f}%)")

    # Apply alpha channel mask (preserves original transparency like rounded corners)
    if alpha_mask is not None:
        if foreground_mask is not None:
            foreground_mask = foreground_mask & alpha_mask
        else:
            foreground_mask = alpha_mask

    # Crop tightly to foreground bounding box for the pipeline output
    if foreground_mask is not None and np.any(foreground_mask):
        rows = np.any(foreground_mask, axis=1)
        cols = np.any(foreground_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Minimal padding (2px) just for anti-aliasing safety
        x_min = max(0, x_min - 2)
        y_min = max(0, y_min - 2)
        x_max = min(img_w - 1, x_max + 2)
        y_max = min(img_h - 1, y_max + 2)

        image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
        foreground_mask = foreground_mask[y_min:y_max + 1, x_min:x_max + 1]
        img_w, img_h = image.size

    # --- Image Info ---
    mm_per_px = target_width / img_w
    st.markdown(
        f"**Output dimensions:** {img_w} × {img_h} px → "
        f"{target_width:.1f} × {img_h * mm_per_px:.1f} mm "
        f"({mm_per_px:.3f} mm/px)"
    )

    # --- Step 1: Color Separation ---
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
        layers = separate_colors(image, config, foreground_mask=foreground_mask)

    # Background layer selection
    auto_bg_idx = next((i for i, l in enumerate(layers) if l.is_background), 0)
    bg_options = {
        i: layer.hex_color for i, layer in enumerate(layers)
    }
    selected_bg = st.radio(
        "Background layer",
        options=list(bg_options.keys()),
        index=auto_bg_idx,
        format_func=lambda i: bg_options[i],
        horizontal=True,
        key="bg_select",
    )

    # Update is_background flags based on selection
    for i, layer in enumerate(layers):
        layer.is_background = (i == selected_bg)

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
            vector_layers = vectorize_layers(
                layers, image.size, config,
                trace_background=foreground_mask is not None,
            )

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

        # 3D Preview (multi-color via GLB + Three.js)
        st.subheader("3D Preview")
        try:
            import trimesh

            combined = trimesh.util.concatenate([ml.mesh for ml in mesh_layers])
            glb_data = combined.export(file_type="glb")
            glb_b64 = base64.b64encode(glb_data).decode()

            threejs_html = f"""
            <div id="viewer" style="width:100%;height:500px;background:#1e1e1e;border-radius:8px;"></div>
            <script type="importmap">
            {{
              "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
              }}
            }}
            </script>
            <script type="module">
            import * as THREE from 'three';
            import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            const container = document.getElementById('viewer');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            scene.add(new THREE.AmbientLight(0xffffff, 1.2));
            const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
            dirLight1.position.set(5, 10, 7);
            scene.add(dirLight1);
            const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
            dirLight2.position.set(-5, 5, -3);
            scene.add(dirLight2);
            const dirLight3 = new THREE.DirectionalLight(0xffffff, 0.4);
            dirLight3.position.set(0, -5, 5);
            scene.add(dirLight3);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const glbBase64 = "{glb_b64}";
            const binary = atob(glbBase64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

            const loader = new GLTFLoader();
            loader.parse(bytes.buffer, '', function(gltf) {{
                const model = gltf.scene;
                // Matte PLA-like finish: high roughness, no metalness
                model.traverse(function(child) {{
                    if (child.isMesh && child.material) {{
                        child.material.roughness = 0.85;
                        child.material.metalness = 0.0;
                    }}
                }});
                scene.add(model);

                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                model.position.sub(center);

                const maxDim = Math.max(size.x, size.y, size.z);
                camera.position.set(maxDim * 0.8, maxDim * 1.0, maxDim * 1.2);
                camera.lookAt(0, 0, 0);
                controls.target.set(0, 0, 0);
                controls.update();
            }});

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            </script>
            """
            components.html(threejs_html, height=520)
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

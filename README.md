---
title: Image to Multi-Color 3D Print
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.54.0"
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
---

# Image to Multi-Color 3D Print (STL/3MF)

Convert any PNG image into multi-color 3D printable files.

## Pipeline

1. **Upload** — PNG/JPG image
2. **Background Removal** — Auto (rembg) + manual refine (magic wand)
3. **Color Separation** — KMeans clustering in CIELAB color space
4. **Vectorize** — Potrace tracing with even-odd fill rule
5. **Extrude** — 3D mesh generation per color layer
6. **Export** — Download as STL, 3MF, or ZIP

## Usage

Upload an image → adjust settings → download 3D files → slice and print!

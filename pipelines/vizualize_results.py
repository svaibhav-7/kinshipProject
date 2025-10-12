#!/usr/bin/env python3
"""
visualize_results.py

Convert the CPU-based pipeline’s .npy outputs into PNG images for sharing.
Usage:
    python pipelines/visualize_results.py --root-dir results_hybrid_features
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import argparse

def save_feature_vector_image(npy_path: Path, out_png: Path, grid_shape=(16, 32)):
    feat = np.load(npy_path)
    assert feat.ndim == 1 and feat.size == grid_shape[0] * grid_shape[1], \
        f"Cannot reshape {feat.size} to {grid_shape}"
    img = feat.reshape(grid_shape)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(str(out_png), img_norm, cmap='viridis')
    print(f"Saved feature visualization: {out_png.name}")

def save_depth_map(npy_path: Path, out_png: Path):
    data = np.load(npy_path)
    depth = data.squeeze()
    assert depth.ndim == 2, f"Depth map must be 2D, got {depth.shape}"
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    plt.imsave(str(out_png), depth_norm, cmap='gray')
    print(f"Saved depth map: {out_png.name}")

def save_mesh_screenshot(ply_path: Path, out_png: Path):
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_png))
    vis.destroy_window()
    print(f"Saved mesh screenshot: {out_png.name}")

def main(root_dir: str):
    root = Path(root_dir)
    for category in root.iterdir():
        if not category.is_dir():
            continue
        print(f"\nProcessing category: {category.name}")
        # combined features
        for npy in category.glob("*_combined_features.npy"):
            base = npy.stem.replace("_combined_features", "")
            save_feature_vector_image(
                npy_path=npy,
                out_png=category / f"{base}_combined_features.png"
            )
        # depth features as grid
        for npy in category.glob("*_depth_features.npy"):
            base = npy.stem.replace("_depth_features", "")
            save_feature_vector_image(
                npy_path=npy,
                out_png=category / f"{base}_depth_features.png",
                grid_shape=(16, 32)
            )
        # synthetic depth map
        for npy in category.glob("*_synthetic_depth.npy"):
            base = npy.stem.replace("_synthetic_depth", "")
            save_depth_map(
                npy_path=npy,
                out_png=category / f"{base}_synthetic_depth.png"
            )
        # 3D mesh
        for ply in category.glob("*_mesh.ply"):
            base = ply.stem.replace("_mesh", "")
            save_mesh_screenshot(
                ply_path=ply,
                out_png=category / f"{base}_mesh.png"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy outputs to PNG images")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject\results_hybrid_features"
    )
    args = parser.parse_args()
    main(args.root_dir)

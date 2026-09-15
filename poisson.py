import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import open3d as o3d  # type: ignore[import-not-found]

from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

# ============================================================
# 1. Load and visualize point cloud
# ============================================================
# def view_3d(filename=None):
#     data = o3d.data.BunnyMesh()

#     mesh = o3d.io.read_triangle_mesh(data.path)

#     pcd = mesh.sample_points_uniformly(number_of_points=50000)

#     return pcd

def view_3d(filename):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Point-cloud file not found: {path}\n"
            "Pass the path to a .ply file, for example: "
            "python 3d-mesh.py C:/data/scan.ply"
        )

    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise ValueError(f"Could not load point cloud: {filename}")

    return pcd



def load_sample_point_cloud():
    dataset = o3d.data.PLYPointCloud()
    return view_3d(dataset.path)


# ============================================================
# 2. Point Cloud -> Mesh
# ============================================================


def point_cloud_to_mesh(
    pcd,
    depth=9,
    scale=1.1,
    linear_fit=False,
    density_quantile_threshold=0.01,
    estimate_normals_if_missing=True
):
    """
    Convert point cloud into a mesh via Screened Poisson Surface Reconstruction:

        Point Cloud
             ↓
        Normal estimation (if missing)
             ↓
        Poisson equation solved over an octree
             ↓
        Implicit function → iso-surface extraction
             ↓
        Density-based trimming (remove low-support extrapolated regions)
             ↓
           Mesh
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud contains no points.")

    # --------------------------------------------------------
    # 1. Poisson reconstruction requires normals. Estimate them
    #    if they're missing (or if caller wants to force it).
    # --------------------------------------------------------
    if estimate_normals_if_missing and (
        not pcd.has_normals() or len(np.asarray(pcd.normals)) != len(points)
    ):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        # Poisson is sensitive to normal orientation consistency
        pcd.orient_normals_consistent_tangent_plane(k=30)

    # --------------------------------------------------------
    # 2. Run screened Poisson reconstruction.
    #    Returns the mesh plus a per-vertex "density" value —
    #    low density = extrapolated / unsupported region.
    # --------------------------------------------------------
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit
    )
    densities = np.asarray(densities)

    # --------------------------------------------------------
    # 3. Trim low-density vertices. Poisson reconstructs a
    #    watertight surface everywhere, including places far
    #    from any input data — those areas have low density
    #    and are usually spurious "bubbles" or blobby extrapolation.
    # --------------------------------------------------------
    if density_quantile_threshold > 0:
        density_cutoff = np.quantile(densities, density_quantile_threshold)
        vertices_to_remove = densities < density_cutoff
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # --------------------------------------------------------
    # 4. Optional: crop the mesh back to the original point
    #    cloud's bounding box, since Poisson can overshoot it.
    # --------------------------------------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # --------------------------------------------------------
    # 5. Cleanup
    # --------------------------------------------------------
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if pcd.has_colors():
        pcd_tree = cKDTree(np.asarray(pcd.points))
        _, idx = pcd_tree.query(np.asarray(mesh.vertices), k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[idx]
        )

    return mesh


# ============================================================
# 3. Automatically estimate parameters
# ============================================================

# def estimate_parameters(
#     pcd,
#     sample_size=10000
# ):

#     points = np.asarray(pcd.points)

#     if len(points) == 0:
#         raise ValueError("Point cloud is empty.")

#     # --------------------------------------------------------
#     # Sample points for faster parameter estimation
#     # --------------------------------------------------------

#     if len(points) > sample_size:

#         indices = np.random.choice(
#             len(points),
#             sample_size,
#             replace=False
#         )

#         sample_points = points[indices]

#     else:
#         sample_points = points

#     # --------------------------------------------------------
#     # KD-tree
#     # --------------------------------------------------------

#     tree = cKDTree(sample_points)

#     # --------------------------------------------------------
#     # Estimate voxel size
#     #
#     # Distance to k nearest neighbors
#     # --------------------------------------------------------

#     k = min(30, len(sample_points) - 1)

#     distances, _ = tree.query(
#         sample_points,
#         k=k + 1
#     )

#     # First column = distance to itself = 0

#     neighbor_distances = distances[:, 1:]

#     avg_distance = np.mean(
#         neighbor_distances
#     )

#     # Heuristic multiplier

#     voxel_size = avg_distance * 2.0

#     # --------------------------------------------------------
#     # Estimate distance distribution
#     # --------------------------------------------------------

#     distances, _ = tree.query(
#         sample_points,
#         k=2
#     )

#     nearest_distances = distances[:, 1:]

#     # --------------------------------------------------------
#     # Coefficient of variation
#     #
#     # CV = standard deviation / mean
#     # --------------------------------------------------------

#     cv = (
#         np.std(nearest_distances)
#         /
#         np.mean(nearest_distances)
#     )

#     # --------------------------------------------------------
#     # Map CV -> percentile
#     # --------------------------------------------------------

#     iso_level_percentile = 50 * (1 - cv)

#     iso_level_percentile = np.clip(
#         iso_level_percentile,
#         10,
#         90
#     )
# 
#       return voxel_size, iso_level_percentile


# ============================================================
# 4. Main pipeline
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create and display a mesh from a PLY point cloud."
    )
    parser.add_argument(
        "filename",
        type=Path,
        nargs="?",
        help="Path to the input PLY point-cloud file"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the browser-based Gradio mesh viewer"
    )
    args = parser.parse_args()

    # if args.web:
    #     create_web_app().launch()
    #     raise SystemExit

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------

    pcd = (
        view_3d(args.filename)
        if args.filename is not None
        else load_sample_point_cloud()
    )

    print(pcd)

    # --------------------------------------------------------
    # Automatically estimate parameters
    # --------------------------------------------------------

    # voxel_size, iso_percentile = estimate_parameters(
    #     pcd
    # )

    # print()
    # print("Estimated parameters:")
    # print("Voxel size:", voxel_size)
    # print("Iso percentile:", iso_percentile)

    # --------------------------------------------------------
    # Point cloud -> mesh
    # --------------------------------------------------------

    mesh = point_cloud_to_mesh(
        pcd
    )

    # --------------------------------------------------------
    # Visualize point cloud and mesh in separate windows
    # --------------------------------------------------------

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud"
    )

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="3D Mesh",
        mesh_show_back_face=True
    )

    # --------------------------------------------------------
    # Save mesh
    # --------------------------------------------------------

    o3d.io.write_triangle_mesh(
        "output_mesh.ply",
        mesh
    )

    print("\nMesh saved to output_mesh.ply")
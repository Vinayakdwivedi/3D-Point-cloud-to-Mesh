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
    voxel_size=0.05,
    iso_level_percentile=50
):
    """
    Convert point cloud into a mesh via true Marching Cubes:

        Point Cloud
             ↓
        3D voxel grid (regular grid over bounding box)
             ↓
        distance field (nearest-point distance at each voxel)
             ↓
        iso-surface value (percentile of distances)
             ↓
        Marching Cubes (skimage)
             ↓
           Mesh
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud contains no points.")

    # --------------------------------------------------------
    # 1. Bounding box, padded so the surface doesn't get clipped
    # --------------------------------------------------------
    mins = np.min(points, axis=0) - voxel_size * 2
    maxs = np.max(points, axis=0) + voxel_size * 2

    dims = np.maximum(
        ((maxs - mins) / voxel_size).astype(int) + 1,
        2
    )

    # Guard against pathologically large grids (memory blow-up)
    max_voxels = 300 ** 3
    if np.prod(dims.astype(np.int64)) > max_voxels:
        scale = (np.prod(dims.astype(np.int64)) / max_voxels) ** (1 / 3)
        voxel_size *= scale
        dims = np.maximum(
            ((maxs - mins) / voxel_size).astype(int) + 1,
            2
        )

    # --------------------------------------------------------
    # 2. Build voxel grid coordinates
    # --------------------------------------------------------
    xs = mins[0] + np.arange(dims[0]) * voxel_size
    ys = mins[1] + np.arange(dims[1]) * voxel_size
    zs = mins[2] + np.arange(dims[2]) * voxel_size

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    # --------------------------------------------------------
    # 3. Unsigned distance field: distance from each voxel
    #    center to the nearest input point
    # --------------------------------------------------------
    tree = cKDTree(points)
    distances, _ = tree.query(grid_points, k=1)
    distance_field = distances.reshape(dims)

    # --------------------------------------------------------
    # 4. Choose iso-level from the distance distribution
    #    (small percentile => surface hugs the points closely)
    # --------------------------------------------------------
    iso_value = np.percentile(distance_field, iso_level_percentile)
    iso_value = max(iso_value, 1e-6)

    # --------------------------------------------------------
    # 5. Marching Cubes on the scalar field
    # --------------------------------------------------------
    verts, faces, normals, _ = marching_cubes(
        distance_field,
        level=iso_value,
        spacing=(voxel_size, voxel_size, voxel_size)
    )

    # Shift vertices back into world coordinates
    verts += mins

    # --------------------------------------------------------
    # 6. Build Open3D mesh from the marching-cubes output
    # --------------------------------------------------------
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

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
        pcd,
        voxel_size=0.01,
        iso_level_percentile=2
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
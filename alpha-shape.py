import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import open3d as o3d  # type: ignore[import-not-found]

from scipy.spatial import cKDTree

# ============================================================
# 1. Load and visualize point cloud
# ============================================================

# def view_3d(filename):
#     path = Path(filename)
#     if not path.is_file():
#         raise FileNotFoundError(
#             f"Point-cloud file not found: {path}\n"
#             "Pass the path to a .ply file, for example: "
#             "python 3d-mesh.py C:/data/scan.ply"
#         )

#     pcd = o3d.io.read_point_cloud(str(path))

#     if pcd.is_empty():
#         raise ValueError(f"Could not load point cloud: {filename}")

#     return pcd
def view_3d(filename=None):
    data = o3d.data.BunnyMesh()

    mesh = o3d.io.read_triangle_mesh(data.path)

    pcd = mesh.sample_points_uniformly(number_of_points=50000)

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
):
    """
    Convert point cloud into a mesh using Open3D's alpha-shape method:

        Point Cloud
             ↓
        3D voxel grid
             ↓
        distance field
             ↓
        iso-surface
             ↓
       Marching Cubes
             ↓
           Mesh
    """

    # --------------------------------------------------------
    # Convert Open3D point cloud -> NumPy
    # --------------------------------------------------------

    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise ValueError("Point cloud contains no points.")

    # --------------------------------------------------------
    # Compute bounding box
    # --------------------------------------------------------

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    # Open3D provides alpha-shape reconstruction without the optional
    # scikit-image dependency or a potentially huge dense voxel array.
    cloud = o3d.geometry.PointCloud(pcd)
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size * 2.0, 1e-6), max_nn=30
            )
        )

    alpha = max(voxel_size * 2.0, 1e-6)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        cloud, alpha
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh

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


    # --------------------------------------------------------
    # Point cloud -> mesh
    # --------------------------------------------------------

    mesh = point_cloud_to_mesh(
        pcd,
        voxel_size=0.009,
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
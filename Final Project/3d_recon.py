import copy
import glob
import os

import cv2
import numpy as np
import open3d as o3d

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DIR = "1_no_obstacle"  # or 0_calib, 1_no_obstacle, 2_obstacle, 3_curved_bend

LEFT_GLOB = f"{DIR}/depth/*.npy"
RIGHT_GLOB = f"{DIR}/depth/*.npy"
RGB_GLOB = f"{DIR}/rectified_left/*.png"

CALIB_FILE = "calibration.npz"

IMAGE_NUMBER = 105
DEPTH_SCALE = 1000.0     # depth in mm -> metres
DEPTH_MIN = 0.5          # metres, clip anything closer than this
DEPTH_MAX = 20.0         # metres, clip anything farther (kills sentinel spikes)
STRIDE = 4               # subsample for faster visualization


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def load_rectified_intrinsics(calib_file=CALIB_FILE):
    """Load P1 (projection matrix of the rectified left camera).

    Depth from stereo is computed on rectified images, so the point cloud
    must be back-projected with P1, not the raw K1.
    P1 is 3x4: [[fx, 0, cx, tx], [0, fy, cy, 0], [0, 0, 1, 0]]
    """
    calib = np.load(calib_file)
    print("calibration.npz keys:", list(calib.keys()))

    P1 = calib["P1"]
    fx, cx = P1[0, 0], P1[0, 2]
    fy, cy = P1[1, 1], P1[1, 2]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1.0]])
    return K, P1


def find_matching_rgb(depth_path):
    """Find the RGB image with the same stem as the depth file."""
    stem = os.path.splitext(os.path.basename(depth_path))[0]
    candidates = sorted(glob.glob(RGB_GLOB))
    for cand in candidates:
        if os.path.splitext(os.path.basename(cand))[0] == stem:
            return cand
    return None


def depth_to_pointcloud(depth, K, rgb=None, depth_scale=DEPTH_SCALE,
                        z_min=DEPTH_MIN, z_max=DEPTH_MAX, stride=STRIDE):
    """Back-project a depth map (HxW) into an Open3D point cloud.

    Invalid / out-of-range depth pixels are discarded — this removes the
    pyramid-with-rays artifact caused by sentinel values.
    If rgb (HxWx3, BGR uint8) is given, per-point colors are set.
    """
    height, width = depth.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth.astype(np.float64) / depth_scale
    mask = (z > z_min) & (z < z_max)

    ys, xs = np.nonzero(mask)
    ys, xs = ys[::stride], xs[::stride]
    z = z[ys, xs]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.stack([x, y, z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if rgb is not None:
        rgb = rgb[:height, :width]  # ensure same resolution
        colors = rgb[ys, xs].astype(np.float64) / 255.0
        # cv2 loads BGR; flip to RGB for Open3D
        colors = colors[:, ::-1]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
    return pcd


def draw_pointcloud(pcd, color=None):
    """Visualize a point cloud in the same style as draw_registrations."""
    temp = copy.deepcopy(pcd)
    if color is not None:
        temp.paint_uniform_color(color)
    o3d.visualization.draw_geometries([temp])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Find depth images
    left_files = sorted(glob.glob(LEFT_GLOB))
    right_files = sorted(glob.glob(RIGHT_GLOB))
    if not left_files:
        raise FileNotFoundError(f"No depth files found for glob: {LEFT_GLOB}")
    print(f"Found {len(left_files)} depth files. Using first: {left_files[IMAGE_NUMBER]}")

    # 2. Load rectified intrinsics (P1) + first depth image
    K, P1 = load_rectified_intrinsics()
    print("Rectified intrinsics from P1:\n", K)

    depth = np.load(left_files[IMAGE_NUMBER])
    print("Depth shape:", depth.shape,
          "min:", depth.min(), "max:", depth.max())

    # 3. RGB image for per-point colors
    rgb_path = find_matching_rgb(left_files[IMAGE_NUMBER])
    rgb = None
    if rgb_path is not None:
        rgb = cv2.imread(rgb_path)
        print("Using RGB image:", rgb_path)
    else:
        print("No matching RGB image found; point cloud stays uncolored.")

    # 4. Back-project to point cloud
    pcd = depth_to_pointcloud(depth, K, rgb=rgb)
    print("Point cloud with", len(pcd.points), "points")

    # 5. Visualize
    draw_pointcloud(pcd)
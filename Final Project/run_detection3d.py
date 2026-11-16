import glob, time
import cv2
import numpy as np
import open3d as o3d

from config import DEPTH_GLOB
from convert3d import load_rectified_intrinsics, find_matching_rgb, depth_to_pointcloud
from detect_background_subtractor import detect_husky


# ----------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------


left_files = sorted(glob.glob(DEPTH_GLOB))
K, P1 = load_rectified_intrinsics()

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()

for i, lf in enumerate(left_files):
    depth = np.load(lf)
    rgb = cv2.imread(find_matching_rgb(lf))

    res = detect_husky(rgb)
    if res is not None:
        x, y, w, h, area, mask = res
        box = np.zeros(mask.shape, bool)
        box[y:y+h, x:x+w] = True
        rgb[(mask == 255) & box] = (0, 0, 255)

    new = depth_to_pointcloud(depth, K, rgb=rgb)
    pcd.points, pcd.colors = new.points, new.colors


    if i == 0:
        vis.add_geometry(pcd)
    else:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)

vis.run()
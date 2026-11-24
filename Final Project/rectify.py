import glob
import os

import cv2
import numpy as np
from config import DIR, LEFT_GLOB, RIGHT_GLOB, CALIB_FILE

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

OUTPUT_LEFT_DIR = f"{DIR}/rectified_left/"
OUTPUT_RIGHT_DIR = f"{DIR}/rectified_right/"

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

os.makedirs(OUTPUT_LEFT_DIR, exist_ok=True)
os.makedirs(OUTPUT_RIGHT_DIR, exist_ok=True)

calibration = np.load(CALIB_FILE)

mtx_left = calibration["K1"]
dist_left = calibration["D1"]
mtx_right = calibration["K2"]
dist_right = calibration["D2"]
R1 = calibration["R1"]
R2 = calibration["R2"]
P1 = calibration["P1"]
P2 = calibration["P2"]
img_size = tuple(calibration["image_size"])  # (width, height) as stored

map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, img_size, cv2.CV_32FC1
)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, img_size, cv2.CV_32FC1
)

def process(glob_pattern, map_x, map_y, out_dir):
    paths = sorted(glob.glob(glob_pattern))
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  failed to read {path}, skipping")
            continue
        rectified = cv2.remap(
            img, map_x, map_y, interpolation=cv2.INTER_LINEAR
        )
        name = os.path.basename(path)
        cv2.imwrite(os.path.join(out_dir, name), rectified)
        print(f"  {path} -> {os.path.join(out_dir, name)}")
    return len(paths)

n_left = process(LEFT_GLOB, map_left_x, map_left_y, OUTPUT_LEFT_DIR)
n_right = process(RIGHT_GLOB, map_right_x, map_y_right := (map_right_x, map_right_y)[1], OUTPUT_RIGHT_DIR) if False else process(RIGHT_GLOB, map_right_x, map_right_y, OUTPUT_RIGHT_DIR)

print(f"done: {n_left} left, {n_right} right frames rectified")
import glob
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DIR = "1_no_obstacle"  # or 0_calib, 1_no_obstacle, 2_obstacle, 3_curved_bend

LEFT_GLOB = f"{DIR}/rectified_left/*.png"
RIGHT_GLOB = f"{DIR}/rectified_right/*.png"

DISP_DIR = f"{DIR}/disparity/"
DEPTH_DIR = f"{DIR}/depth/"

CALIB_FILE = "calibration.npz"

# ----------------------------------------------------------------------------
# DISPARITY
# ----------------------------------------------------------------------------

def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return cv2.GaussianBlur(g, (5, 5), 0)


def make_disparity(rect_left, rect_right, wsize=11):
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=wsize,
        P1=8 * 3 * wsize ** 2,
        P2=64 * 3 * wsize ** 2,
        disp12MaxDiff=2,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=1,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
    )
    prep(rect_left)
    disp = matcher.compute(prep(rect_left), prep(rect_right)).astype(np.float32) / 16.0
    return disp  # disparity in pixels, negative = invalid

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

os.makedirs(DISP_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

calibration = np.load(CALIB_FILE)
baseline = float(np.linalg.norm(calibration["T"].ravel()))
focal = float(calibration["P1"][0, 0])

left_paths = sorted(glob.glob(LEFT_GLOB))
right_paths = sorted(glob.glob(RIGHT_GLOB))

if len(left_paths) != len(right_paths):
    raise ValueError(
        f"pair count mismatch: {len(left_paths)} left vs {len(right_paths)} right"
    )

for left_path, right_path in zip(left_paths, right_paths):
    name = os.path.basename(left_path)

    rect_left = cv2.imread(left_path)
    rect_right = cv2.imread(right_path)
    if rect_left is None or rect_right is None:
        print(f"  failed to read pair {name}, skipping")
        continue

    disp = make_disparity(rect_left, rect_right)
    depth = np.where(disp > 0, focal * baseline / np.maximum(disp, 1e-6), 0)

    np.save(os.path.join(DISP_DIR, name.replace(".png", ".npy")), disp)
    np.save(os.path.join(DEPTH_DIR, name.replace(".png", ".npy")), depth)

    # visual check: normalized disparity image
    valid = disp > 0
    vmax = np.percentile(disp[valid], 98)
    disp_vis = (np.clip(disp, 0, vmax) / vmax * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DISP_DIR, name), disp_vis)

    print(f"  {name}: disp saved")

print(f"done: {len(left_paths)} pairs processed")
"""
Detect the Husky with dense optical flow (Farneback), left camera only.

The stereo rig does not move, so every pixel with real motion should belong to
the robot. Dense flow gives a motion vector for every pixel, not only for corners,
so the dark, low-texture body and wheels are covered too.

    1. Compute dense flow between frame t and t+1 (on a downscaled image for speed)
    2. Threshold the flow magnitude -> motion mask
    3. Clean the mask with morphology (remove specks, fill holes)
    4. Take the largest blob as the robot, draw its bounding box

Run rectify.py first. This script reads its rectified_left/ output.
Output: drawings in OUTPUT_DIR and all boxes in OUTPUT_DIR/boxes.npy
(one row per frame: frame index, x, y, w, h; x = -1 when nothing was found).
"""

import glob
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DIR = "3_curved_bend" #"2_obstacle" #"1_no_obstacle"  # or 1_no_obstacle, 2_obstacle, 3_curved_bend

LEFT_GLOB = f"{DIR}/rectified_left/*.png"
OUTPUT_DIR = f"{DIR}/detect_dense/"

SCALE = 0.5            # flow is computed at this image scale (Farneback is slow at full size)
MIN_MOTION_PX = 1.0    # flow magnitude (in full-size pixels) above which a pixel counts as moving
MIN_AREA_PX = 1500     # blobs smaller than this (full-size pixels) are noise
OPEN_KERNEL = 5        # removes small specks
CLOSE_KERNEL = 25      # joins the robot's parts into one blob

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# ----------------------------------------------------------------------------
# DENSE FLOW
# ----------------------------------------------------------------------------

def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0)


def motion_mask(gray1, gray2):
    """Returns the full-size binary motion mask and the flow magnitude."""
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **FARNEBACK_PARAMS)
    mag = np.linalg.norm(flow, axis=2) / SCALE  # back to full-size pixels
    mask = (mag > MIN_MOTION_PX).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8))

    full_size = (int(round(gray1.shape[1] / SCALE)), int(round(gray1.shape[0] / SCALE)))
    mask = cv2.resize(mask, full_size, interpolation=cv2.INTER_NEAREST)
    mag = cv2.resize(mag, full_size, interpolation=cv2.INTER_LINEAR)
    return mask, mag


def detect(mask):
    """Bounding box (x, y, w, h) of the largest moving blob, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA_PX]
    if not contours:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def draw(img, mask, box):
    out = img.copy()
    overlay = out.copy()
    overlay[mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)  # moving pixels tinted green
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = sorted(glob.glob(LEFT_GLOB))
if len(paths) < 2:
    raise SystemExit(f"need at least 2 images in {LEFT_GLOB} (run rectify.py first)")

prev_gray = prep(cv2.imread(paths[0]))
boxes = []

for i, path in enumerate(paths[1:], start=1):
    img = cv2.imread(path)
    if img is None:
        print(f"  failed to read {path}, skipping")
        continue
    gray = prep(img)

    mask, mag = motion_mask(prev_gray, gray)
    mask = mask[: img.shape[0], : img.shape[1]]
    box = detect(mask)
    boxes.append((i, *box) if box is not None else (i, -1, -1, -1, -1))

    name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), draw(img, mask, box))
    print(f"  {name}: {int((mask > 0).sum())} moving px, box={box}")

    prev_gray = gray

boxes = np.array(boxes, dtype=np.int32)
np.save(os.path.join(OUTPUT_DIR, "boxes.npy"), boxes)
found = int((boxes[:, 1] >= 0).sum())
print(f"done: robot found in {found}/{len(boxes)} frames, images in {OUTPUT_DIR}")

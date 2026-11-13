"""
Detect the Husky with sparse optical flow (Lucas-Kanade), left camera only.

The stereo rig does not move, so the background stays still and the features
that move between two frames should belong to the robot.

    1. Find good corners in frame t
    2. Track them into frame t+1 with calcOpticalFlowPyrLK
    3. Keep the points that tracked and moved more than MIN_MOTION_PX
    4. Draw the flow vectors and a bounding box around the moving points

Run rectify.py first. This script reads its rectified_left/ output.
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
OUTPUT_DIR = f"{DIR}/detect_flow/"

MAX_CORNERS = 300
QUALITY_LEVEL = 0.01   # 0.3 keeps only very strong corners, often too few on the robot
MIN_DISTANCE = 5
MIN_MOTION_PX = 1.5    # flow shorter than this counts as noise / static background
MIN_MOVING_POINTS = 5  # fewer moving points than this -> no detection in this frame

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# ----------------------------------------------------------------------------
# OPTICAL FLOW
# ----------------------------------------------------------------------------

def flow_between(gray1, gray2):
    """Track corners from gray1 to gray2. Returns (start points, end points), both Nx2."""
    feat1 = cv2.goodFeaturesToTrack(
        gray1, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE
    )
    if feat1 is None:
        return np.empty((0, 2)), np.empty((0, 2))
    feat2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None, **LK_PARAMS)
    ok = status.ravel() == 1  # drop points LK lost
    return feat1.reshape(-1, 2)[ok], feat2.reshape(-1, 2)[ok]


def detect(p1, p2):
    """Bounding box (x, y, w, h) around the moving points, or None."""
    motion = np.linalg.norm(p2 - p1, axis=1)
    moving = p2[motion > MIN_MOTION_PX]
    if len(moving) < MIN_MOVING_POINTS:
        return None, moving
    x, y, w, h = cv2.boundingRect(moving.astype(np.float32))
    return (x, y, w, h), moving


def draw(img, p1, p2, box):
    out = img.copy()
    motion = np.linalg.norm(p2 - p1, axis=1)
    for (x1, y1), (x2, y2), m in zip(p1.astype(int), p2.astype(int), motion):
        color = (0, 255, 0) if m > MIN_MOTION_PX else (120, 120, 120)  # green = moving
        cv2.line(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (x2, y2), 3, color, -1)
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

prev_img = cv2.imread(paths[0])
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
detections = []

for path in paths[1:]:
    img = cv2.imread(path)
    if img is None:
        print(f"  failed to read {path}, skipping")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p1, p2 = flow_between(prev_gray, gray)
    box, moving = detect(p1, p2)
    detections.append(box)

    name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), draw(img, p1, p2, box))
    print(f"  {name}: {len(p1)} tracked, {len(moving)} moving, box={box}")

    prev_gray = gray

found = sum(b is not None for b in detections)
print(f"done: robot found in {found}/{len(detections)} frames, images in {OUTPUT_DIR}")

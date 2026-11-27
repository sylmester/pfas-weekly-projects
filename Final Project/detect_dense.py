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
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DIR = sys.argv[1] if len(sys.argv) > 1 else "3_curved_bend"  # or 1_no_obstacle, 2_obstacle, 3_curved_bend

LEFT_GLOB = f"{DIR}/rectified_left/*.png"
OUTPUT_DIR = f"{DIR}/detect_dense/"

SCALE = 0.5            # flow is computed at this image scale (Farneback is slow at full size)
MIN_MOTION_PX = 1.0    # flow magnitude (in full-size pixels) above which a pixel counts as moving
MIN_AREA_PX = 1500     # blobs smaller than this (full-size pixels) are noise
OPEN_KERNEL = 5        # removes small specks
CLOSE_KERNEL = 25      # joins the robot's parts into one blob

# Farneback reports spurious flow on dark, textureless areas near the image border
# (e.g. the bottom-right floor corner in 2_obstacle) although the pixels barely change.
# A blob only counts if the mean grey-level change between the two frames inside it is
# above this. Measured on 2_obstacle: false blobs <= 1.3, real robot >= 4.1.
MIN_INTENSITY_DIFF = 2.5

# The robot drives on the floor, so it never touches the top of the image. Blobs that do
# are the thin top-edge glitch strips (2_obstacle frames 3-4, 3_curved_bend frames 87-88).
TOP_MARGIN_PX = 5

# When there are several blobs, pick the one closest to the last detection instead of the
# largest, as long as that detection is at most this many frames old. Otherwise: largest.
MAX_FRAMES_SINCE_LAST = 10

# --- Gap filling for frames where the robot is visible but not moving --------------------
# Optical flow only sees motion. When the robot stands still (e.g. the first ~40 frames
# of 3_curved_bend, or when it stops to turn) we follow it with template matching from
# the nearest flow detection, frame by frame, forward and backward.
SEARCH_MARGIN_PX = 30    # search window around the previous box
MIN_MATCH_SCORE = 0.6    # TM_CCOEFF_NORMED threshold for accepting a template match
# A match also has to differ from the empty scene, otherwise the tracker could lock onto
# the background once the robot has left / is hidden. Background = per-pixel median over
# the sequence (the robot moves, so the median is the empty scene).
MIN_BG_DIFF = 15.0       # mean |patch - background| grey levels inside the box
BG_STEP = 5              # use every BG_STEP-th frame for the median
BORDER_PX = 3            # don't start a template from a box cut off by the image border

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


def intensity_change(contour, diff):
    """Mean absolute grey-level difference between the two frames inside the blob."""
    blob = np.zeros(diff.shape, np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, -1)
    return float(diff[blob > 0].mean())


def box_center(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2])


def detect(mask, diff, last_box=None):
    """Bounding box (x, y, w, h) of the real moving blob, or None.

    diff: full-size absolute difference of the two grey frames, used to reject
    blobs where the flow says "moving" but the pixels did not actually change.
    last_box: a recent detection. If given, the blob closest to it is chosen,
    otherwise the largest blob.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        c for c in contours
        if cv2.contourArea(c) >= MIN_AREA_PX and intensity_change(c, diff) >= MIN_INTENSITY_DIFF
    ]
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    candidates = [(b, a) for b, a in zip(boxes, areas) if b[1] > TOP_MARGIN_PX]
    if not candidates:
        return None
    if last_box is None:
        return max(candidates, key=lambda ba: ba[1])[0]
    last = box_center(last_box)
    return min(candidates, key=lambda ba: np.linalg.norm(box_center(ba[0]) - last))[0]


def touches_border(box, shape):
    x, y, w, h = box
    return x <= BORDER_PX or y <= BORDER_PX or x + w >= shape[1] - BORDER_PX or y + h >= shape[0] - BORDER_PX


def match_template(gray, template, prev_box, background):
    """Find template near prev_box in gray. Returns (box, score, bg_diff) or None."""
    x, y, w, h = prev_box
    H, W = gray.shape
    x0, y0 = max(0, x - SEARCH_MARGIN_PX), max(0, y - SEARCH_MARGIN_PX)
    x1, y1 = min(W, x + w + SEARCH_MARGIN_PX), min(H, y + h + SEARCH_MARGIN_PX)
    window = gray[y0:y1, x0:x1]
    if window.shape[0] < h or window.shape[1] < w:
        return None
    res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    box = (x0 + mx, y0 + my, w, h)
    bx, by = box[0], box[1]
    bg_diff = float(np.abs(gray[by:by + h, bx:bx + w].astype(np.float32)
                           - background[by:by + h, bx:bx + w]).mean())
    return box, score, bg_diff


def track_through_gap(frames, start_idx, start_box, step, stop_idx, boxes, background, gray_at):
    """Follow the robot from a flow detection at start_idx in direction step (+1/-1)
    until stop_idx (exclusive) or the match is no longer trusted. Fills boxes in place."""
    if touches_border(start_box, background.shape):
        return
    prev_gray, prev_box = gray_at(frames[start_idx]), start_box
    i = start_idx + step
    while i != stop_idx and boxes[i] is None:
        gray = gray_at(frames[i])
        x, y, w, h = prev_box
        template = prev_gray[y:y + h, x:x + w]
        m = match_template(gray, template, prev_box, background)
        if m is None or m[1] < MIN_MATCH_SCORE or m[2] < MIN_BG_DIFF:
            if m is not None:
                print(f"  frame {i}: template stop (score={m[1]:.2f}, bg_diff={m[2]:.1f})")
            return
        boxes[i] = m[0]
        prev_gray, prev_box = gray, m[0]
        i += step


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

prev_img = cv2.imread(paths[0])
prev_gray = prep(prev_img)
prev_full = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
flow_boxes = [None] * len(paths)  # index = frame index in paths; frame 0 has no flow
last_box, last_frame = None, None  # most recent detection, used to choose between blobs

for i, path in enumerate(paths[1:], start=1):
    img = cv2.imread(path)
    if img is None:
        print(f"  failed to read {path}, skipping")
        continue
    gray = prep(img)
    full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask, mag = motion_mask(prev_gray, gray)
    mask = mask[: img.shape[0], : img.shape[1]]
    recent = last_box if last_frame is not None and i - last_frame <= MAX_FRAMES_SINCE_LAST else None
    box = detect(mask, cv2.absdiff(full, prev_full), recent)
    if box is not None:
        last_box, last_frame = box, i
    flow_boxes[i] = box

    name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), draw(img, mask, box))
    print(f"  {name}: {int((mask > 0).sum())} moving px, box={box}")

    prev_gray = gray
    prev_full = full

# ----------------------------------------------------------------------------
# GAP FILLING (robot visible but standing still)
# ----------------------------------------------------------------------------

def gray_at(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

print("building background (median) ...")
background = np.median(
    np.stack([gray_at(p) for p in paths[::BG_STEP]]), axis=0
).astype(np.float32)

boxes = list(flow_boxes)
detected = [i for i, b in enumerate(flow_boxes) if b is not None]
for k, i in enumerate(detected):
    # forward from this detection up to the next one, then backward from the next one
    nxt = detected[k + 1] if k + 1 < len(detected) else len(paths)
    track_through_gap(paths, i, flow_boxes[i], +1, nxt, boxes, background, gray_at)
    prv = detected[k - 1] if k > 0 else -1
    track_through_gap(paths, i, flow_boxes[i], -1, prv, boxes, background, gray_at)

# redraw frames filled by template matching (blue box instead of red)
filled = [i for i in range(len(paths)) if boxes[i] is not None and flow_boxes[i] is None]
for i in filled:
    name = os.path.basename(paths[i])
    out_path = os.path.join(OUTPUT_DIR, name)
    img = cv2.imread(out_path) if os.path.exists(out_path) else cv2.imread(paths[i])
    x, y, w, h = boxes[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(out_path, img)

rows = [(i, *b) if b is not None else (i, -1, -1, -1, -1) for i, b in enumerate(boxes)]
boxes = np.array(rows, dtype=np.int32)
np.save(os.path.join(OUTPUT_DIR, "boxes.npy"), boxes)
found = int((boxes[:, 1] >= 0).sum())
print(f"done: robot found in {found}/{len(boxes)} frames "
      f"({len(detected)} by flow, {len(filled)} by template), images in {OUTPUT_DIR}")

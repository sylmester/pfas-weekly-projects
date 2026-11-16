# %% [markdown]
# # Husky detection
# 
# Detects the Husky A200 in every rectified left frame and computes its 3D position from the disparity map.
# 
# Pipeline per frame:
# 1. **Background**: median of 60 frames of the sequence (the robot moves, so the median is the empty scene).
# 2. **Foreground mask**: `|frame - background|` > threshold, cleaned with opening + closing.
# 3. **Candidates**: connected components of the mask.
# 4. **3D position**: median disparity over the blob pixels -> `Z = f·B/d`, `X = (u-cx)·Z/f`, `Y = (v-cy)·Z/f`.
# 5. **Is it the Husky?** Pixel size -> real size with the depth (`size_m = size_px·Z/f`) and check it against the robot dimensions.
# 
# Output per sequence: `{seq}/detections.csv` (input of the tracking step), `{seq}/detection/*.jpg` (annotated frames for the video) and `{seq}/background.png`.
# 
# Requires `rectify.ipynb` and `disparityMap.ipynb` to have been run first.

# %%
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
SEQUENCES = ["1_no_obstacle", "2_obstacle", "3_curved_bend", "4_circle"]
CALIB_FILE = "calibration.npz"

BG_FRAMES = 60            # frames sampled to build the median background
DIFF_THRESH = 30          # grey-level change that counts as foreground
OPEN_KERNEL = 3           # px, removes isolated noise pixels
CLOSE_KERNEL = 15         # px, fills holes inside the robot / joins its parts
MIN_AREA = 300            # px; the robot at 18 m is still ~1000 px

# Husky A200 body ~0.99 x 0.67 x 0.39 m, plus a sensor mast on top.
# Measured in the images: ~0.8-1.2 m wide and ~1.1-1.35 m tall (with the mast).
# The lower width limit is small so a robot that is only partly visible
# (entering the image, behind the obstacle) is still accepted, flagged as 'partial'.
WIDTH_RANGE_M = (0.25, 1.8)
HEIGHT_RANGE_M = (0.3, 1.6)
FULL_WIDTH_M = 0.6        # narrower than this -> only part of the robot is visible
MAX_DEPTH_M = 25.0        # the arena is not deeper than this

# %% [markdown]
# ## Functions

# %%
def load_camera(calib_file=CALIB_FILE):
    """Rectified left camera: focal length f, principal point (cx, cy), baseline B (m)."""
    c = np.load(calib_file)
    P1 = c["P1"]
    return dict(f=float(P1[0, 0]), cx=float(P1[0, 2]), cy=float(P1[1, 2]),
                B=float(np.linalg.norm(c["T"])) / 1000.0)


def build_background(paths, n=BG_FRAMES):
    """Median of n frames spread over the sequence.
    The robot moves, so at every pixel most frames see the background -> the median
    is an image of the empty scene. Static obstacles (the board in sequence 2) stay in it."""
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    stack = np.stack([cv2.imread(paths[i]) for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


def foreground_mask(frame, background, thresh=DIFF_THRESH):
    """Pixels that differ from the background, cleaned with morphology."""
    diff = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 0), cv2.GaussianBlur(background, (5, 5), 0))
    diff = diff.max(axis=2)                       # strongest change over the B, G, R channels
    mask = (diff > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL)))
    return mask


def find_candidates(mask, min_area=MIN_AREA):
    """Connected blobs of the mask -> list of dicts with bbox, area, centroid and pixel mask."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, n):                         # label 0 is the background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        candidates.append(dict(bbox=(int(x), int(y), int(w), int(h)), area=int(area),
                               u=float(centroids[i][0]), v=float(centroids[i][1]),
                               pixels=labels == i))
    return candidates


def locate_3d(cand, disparity, cam):
    """Median disparity over the blob's own pixels (not the bbox, which also contains
    far background with d ~ 3 px) -> X, Y, Z in metres in the rectified left camera
    frame (X right, Y down, Z forward)."""
    d = disparity[cand["pixels"]]
    d = d[d > 0]
    if len(d) < 10:
        return None
    d = float(np.median(d))
    if d < cam["f"] * cam["B"] / MAX_DEPTH_M:     # further than the arena -> wrong match
        return None
    Z = cam["f"] * cam["B"] / d
    X = (cand["u"] - cam["cx"]) * Z / cam["f"]
    Y = (cand["v"] - cam["cy"]) * Z / cam["f"]
    return dict(disp=d, X=X, Y=Y, Z=Z)


def real_size(cand, Z, cam):
    """Pixel size -> metres: size_m = size_px * Z / f (pinhole model)."""
    _, _, w, h = cand["bbox"]
    return w * Z / cam["f"], h * Z / cam["f"]


def classify(cand, cam, image_width):
    """Sets cand['valid'] (size compatible with the Husky) and cand['partial']
    (touches the image border or too narrow -> only part of the robot is visible,
    so its centroid is not the robot centre)."""
    cand["valid"] = cand["partial"] = False
    if cand["pos"] is None:
        return
    w_m, h_m = cand["size_m"] = real_size(cand, cand["pos"]["Z"], cam)
    cand["valid"] = (WIDTH_RANGE_M[0] <= w_m <= WIDTH_RANGE_M[1]
                     and HEIGHT_RANGE_M[0] <= h_m <= HEIGHT_RANGE_M[1])
    x, _, w, _ = cand["bbox"]
    cand["partial"] = x <= 1 or x + w >= image_width - 1 or w_m < FULL_WIDTH_M


def detect_frame(frame, disparity, background, cam):
    """Returns (detection or None, mask, all candidates). If several blobs are valid,
    the largest one wins (the tracking step will use the predicted position instead)."""
    mask = foreground_mask(frame, background)
    candidates = find_candidates(mask)
    for c in candidates:
        c["pos"] = locate_3d(c, disparity, cam)
        classify(c, cam, frame.shape[1])
    valid = [c for c in candidates if c["valid"]]
    best = max(valid, key=lambda c: c["area"]) if valid else None
    return best, mask, candidates


def draw_detection(frame, det, candidates, text=""):
    """Rejected blobs in grey, the detection in green (orange if partial) with its 3D position."""
    out = frame.copy()
    for c in candidates:
        if c is not det:
            x, y, w, h = c["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (160, 160, 160), 1)
    if det is not None:
        x, y, w, h = det["bbox"]
        p = det["pos"]
        color = (0, 165, 255) if det["partial"] else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (int(det["u"]), int(det["v"])), 4, (0, 0, 255), -1)
        label = f"X={p['X']:+.2f} Y={p['Y']:+.2f} Z={p['Z']:.2f} m" + (" (partial)" if det["partial"] else "")
        cv2.putText(out, label, (max(0, min(x, out.shape[1] - 330)), max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


CSV_HEADER = "frame,detected,partial,x,y,w,h,u,v,disp,X,Y,Z"


def run_sequence(seq, cam, save_images=True):
    """Detect the Husky in every frame of a sequence.
    Writes {seq}/detections.csv, {seq}/background.png and {seq}/detection/*.jpg."""
    paths = sorted(glob.glob(f"{seq}/rectified_left/*.png"))
    background = build_background(paths)
    cv2.imwrite(f"{seq}/background.png", background)
    if save_images:
        os.makedirs(f"{seq}/detection", exist_ok=True)

    rows = []
    for i, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        frame = cv2.imread(path)
        disparity = np.load(f"{seq}/disparity/{stem}.npy")
        det, _, candidates = detect_frame(frame, disparity, background, cam)
        if det is None:
            rows.append([i, 0, 0] + [np.nan] * 10)
        else:
            p = det["pos"]
            rows.append([i, 1, int(det["partial"]), *det["bbox"], det["u"], det["v"],
                         p["disp"], p["X"], p["Y"], p["Z"]])
        if save_images:
            cv2.imwrite(f"{seq}/detection/{stem}.jpg", draw_detection(frame, det, candidates, f"{seq}  frame {i}"))

    rows = np.array(rows, dtype=float)
    np.savetxt(f"{seq}/detections.csv", rows, delimiter=",", header=CSV_HEADER, comments="",
               fmt=["%d", "%d", "%d"] + ["%.4f"] * 10)
    return rows

# %% [markdown]
# ## 1. Test on a single frame
# Change `SEQ` / `FRAME` to inspect any frame: background, mask and final detection.

# %%
SEQ = "2_obstacle"
FRAME = 120

cam = load_camera()
print("camera:", {k: round(v, 3) for k, v in cam.items()})

paths = sorted(glob.glob(f"{SEQ}/rectified_left/*.png"))
background = build_background(paths)
frame = cv2.imread(paths[FRAME])
disparity = np.load(f"{SEQ}/disparity/{FRAME:06d}.npy")
det, mask, candidates = detect_frame(frame, disparity, background, cam)

for c in candidates:
    size = tuple(round(s, 2) for s in c.get("size_m", ()))
    z = None if c["pos"] is None else round(c["pos"]["Z"], 2)
    print(f"blob bbox={c['bbox']} area={c['area']} Z={z} m size={size} m valid={c['valid']} partial={c['partial']}")

fig, ax = plt.subplots(2, 2, figsize=(16, 9))
ax[0, 0].imshow(background[..., ::-1]); ax[0, 0].set_title("background (median)")
ax[0, 1].imshow(mask, cmap="gray"); ax[0, 1].set_title("foreground mask")
ax[1, 0].imshow(np.clip(disparity, 0, 60), cmap="jet"); ax[1, 0].set_title("disparity")
ax[1, 1].imshow(draw_detection(frame, det, candidates)[..., ::-1]); ax[1, 1].set_title("detection")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Run all sequences
# Saves the CSV and the annotated images (a few minutes for the four sequences).

# %%
results = {}
for seq in SEQUENCES:
    rows = run_sequence(seq, cam, save_images=True)
    results[seq] = rows
    det = rows[:, 1] == 1
    print(f"{seq}: detected in {det.sum()}/{len(rows)} frames ({int(rows[det, 2].sum())} partial) | "
          f"Z {np.nanmin(rows[:, 12]):.2f} - {np.nanmax(rows[:, 12]):.2f} m")

# %% [markdown]
# ## 3. Detected positions
# Left: X and Z over time (gaps = no detection, e.g. the occlusion in sequence 2). Right: top view (X–Z) of the raw detections, before any tracking/filtering.

# %%
fig, ax = plt.subplots(len(SEQUENCES), 2, figsize=(14, 4 * len(SEQUENCES)))
for k, seq in enumerate(SEQUENCES):
    r = results[seq]
    full, part = (r[:, 1] == 1) & (r[:, 2] == 0), r[:, 2] == 1
    ax[k, 0].plot(r[:, 0], r[:, 10], ".", ms=3, label="X")
    ax[k, 0].plot(r[:, 0], r[:, 12], ".", ms=3, label="Z")
    ax[k, 0].set_title(seq); ax[k, 0].set_xlabel("frame"); ax[k, 0].set_ylabel("m"); ax[k, 0].legend()
    ax[k, 1].plot(r[full, 10], r[full, 12], ".", ms=3, label="full")
    ax[k, 1].plot(r[part, 10], r[part, 12], ".", ms=3, color="orange", label="partial")
    ax[k, 1].plot(0, 0, "k^", ms=10, label="camera")
    ax[k, 1].set_xlabel("X [m]"); ax[k, 1].set_ylabel("Z [m]"); ax[k, 1].set_aspect("equal", "datalim")
    ax[k, 1].legend()
plt.tight_layout()
plt.show()



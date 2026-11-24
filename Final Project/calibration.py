"""
Stereo camera calibration (left + right) from chessboard images.

    1. Detect the chessboard in every left/right image (several detectors x several scales)
    2. Calibrate each camera on ALL of its own detections (left and right)
    3. Calibrate the stereo geometry (R, T) on the image pairs found in both cameras
    4. Rectify, measure the alignment, save the result

Debug output (folder DEBUG_DIR):
    detections/  every image pair with the detected corners drawn (colour runs red -> blue in
                 corner order, so a wrong/flipped order is visible at a glance)
    stages/      a few pairs shown after each processing stage, with epipolar lines
Units: millimetres for the board and translations, pixels for the image.
"""

import glob
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
LEFT_GLOB = "0_calib/left/*.png"
RIGHT_GLOB = "0_calib/right/*.png"
PATTERN = (7, 5)
SQUARE_SIZE_MM = 80.0

# When Calibrating the intrinsic camera model for each camera it warped dramatically at the edges
# Keeping the k3 fixed at 0 removes the radial distortion that is strongest at the edges
INTRINSIC_FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

# Image scales tried by the corner detector. See find_corners().
SCALES = (1.0, 0.5, 0.25, 1.25, 1.5, 2.0, 2.5)

ALPHA = 0                     # final rectification: 0 = crop to valid pixels, 1 = keep every pixel
PREVIEW_PAIRS = 5             # how many pairs to render in the stage previews
OUTPUT_FILE = "calibration.npz"
DEBUG_DIR = "calib_debug"

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]

# Same board for both cameras -> one set of object points, in mm
OBJP = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
OBJP[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

# ----------------------------------------------------------------------------
# GLOSSARY: every notation that appears in the printouts / saved file
# ----------------------------------------------------------------------------
GLOSSARY = {
    "view / pair": "one image of one camera / one left+right image pair taken at the same time",
    "corner": "inner chessboard corner (4 squares meet). PATTERN=(7,5) means 7 columns x 5 rows of them",
    "SB / classic": "the two corner detectors: findChessboardCornersSB (sector-based) / findChessboardCorners",
    "xN.N": "image scale used when the corners were found, e.g. 'SB x0.5' = sector detector on half-size image",
    "objp": "known 3D position of every board corner in mm (same board for every view)",
    "image points": "pixel position (x, y) where a corner was detected",
    "RMS": "root-mean-square reprojection error in pixels: how far, on average, the model's predicted "
           "corner positions are from the detected ones. Lower is better (~0.3-0.5 is good)",
    "K": "camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]: the camera's internal geometry (no lens distortion)",
    "fx, fy": "focal length in pixels (horizontal / vertical)",
    "cx, cy": "principal point in pixels: where the optical axis hits the image (should be near width/2, height/2)",
    "D": "lens distortion coefficients (k1, k2, p1, p2, k3)",
    "k1, k2, k3": "radial distortion: straight lines bending towards/away from the image centre",
    "p1, p2": "tangential distortion: lens not parallel to the sensor (zeroed by ZERO_TANGENT_DIST)",
    "R, T": "pose of the right camera relative to the left: X_right = R @ X_left + T. T is in mm",
    "rotation [deg]": "R written as rotation about the x, y, z axes of the left camera",
    "baseline": "distance between the two cameras = length of T, in mm",
    "E": "essential matrix: encodes R and T; x_right^T E x_left = 0 for matching normalised points",
    "F": "fundamental matrix: same as E but for pixel coordinates",
    "epipolar line": "for a point in the left image, the line in the right image on which its match must lie "
                     "(a curve in a distorted image, straight once undistorted, horizontal once rectified)",
    "rectified": "both images warped so epipolar lines become horizontal rows with identical y in both images",
    "R1, R2": "rotations that rectify the left / right image",
    "P1, P2": "new projection matrices of the rectified left / right camera",
    "Q": "matrix converting disparity to 3D (use with cv2.reprojectImageTo3D)",
    "alpha": "rectification zoom: 0 = crop to valid pixels only, 1 = keep all pixels (black borders)",
    "ROI": "(x, y, width, height) rectangle of the rectified image that contains only valid pixels",
    "vertical misalignment": "|y_left - y_right| of matching corners after rectification; ideally 0",
}


# ----------------------------------------------------------------------------
# STEP 1: CORNER DETECTION
# ----------------------------------------------------------------------------

def refine(gray, corners):
    """
    So when we find corners it will not be possible to find a pixel that is in the exact corner position.
    We therefore do some "sub-pixel refinement" that looks at a number of pixels around the corner position.
    It can use the gradients from a search window to estimate the true corner position.

    Now the chessboard is positioned at many different distances during the calibration, so the pixel size of the 
    board will also change significantly.

    The search window must stay close to the corner, so it is derived from the square size 
    (a fixed 11 px window drifts on small boards).

    """

    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
    
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2) # finds the horizontal distance between corners in pixels
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2) # --vertical--
    square_px = min(np.median(dx), np.median(dy)) # Take the smallest distance.

    half_win = int(0.45 * square_px)
    refined = cv2.cornerSubPix(gray, corners, (half_win, half_win), (-1, -1), SUBPIX_CRITERIA)
    return refined.reshape(-1, 2)  



def find_corners(gray):
    """
    Try both detectors on several rescaled copies of the image, return the first success.

    Detector types: 
        SB:         (Sector-based) works basically by taking the average colour of lines of different angles.
                    A window around a corner is divided into 4 sectors. If the line runs along the grid it will be
                    switch color halfway makign the average gray. However if the line runsdiagonally it will run
                    primarily along one color and therefore be at the endpoints instead.

        classic:    thresholds the image, finds the black squares as quadrilaterals and links
                    them into a grid. Needs squares of roughly 10+ px that stay separated from
                    each other and from other dark objects.
        
    Success depends on the scale non-monotonically (it is not 'bigger is better'), so we
    simply try many scales: downscaling suppresses blur/noise, upscaling helps tiny boards.
    """

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    for name in ("SB", "classic"):
        for scale in SCALES:

            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC # define the interpolation
            img = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp) # Scale image

            if name == "SB": # Sector-based
                ok, c = cv2.findChessboardCornersSB(img, PATTERN, sb_flags)
            else:
                ok, c = cv2.findChessboardCorners(img, PATTERN, classic_flags)
            if ok:
                return refine(gray, (c / scale).astype(np.float32)), f"{name} x{scale}"

    # If nothing works return non found
    return None, "not found"


def align_orientation(corners_left, corners_right):
    """
    An 8x6-square board looks identical rotated 180 deg, so the detector may number the corners
    from opposite ends in the two images. Assumes a roughly parallel, side-by-side rig.
    Returns (corners_right in the left camera's order, was_flipped).
    """
    v_left = corners_left[-1, 0] - corners_left[0, 0]
    v_right = corners_right[-1, 0] - corners_right[0, 0]
    if np.dot(v_left, v_right) < 0:
        return corners_right[::-1].copy(), True
    return corners_right, False


def detect_all():
    """Detect every pair, save annotated images. Returns (list of view dicts, image_size)."""
    left_files, right_files = sorted(glob.glob(LEFT_GLOB)), sorted(glob.glob(RIGHT_GLOB))
    if not left_files or len(left_files) != len(right_files):
        raise RuntimeError(f"Found {len(left_files)} left and {len(right_files)} right images")
    os.makedirs(f"{DEBUG_DIR}/detections", exist_ok=True)

    views = []
    for lf, rf in zip(left_files, right_files):
        name = os.path.basename(lf)
        img_l, img_r = cv2.imread(lf), cv2.imread(rf)
        if img_l is None or img_r is None or img_l.shape != img_r.shape:
            raise RuntimeError(f"Cannot read, or size mismatch: {lf} / {rf}")
        image_size = img_l.shape[1::-1]  # (width, height)

        c_l, how_l = find_corners(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
        c_r, how_r = find_corners(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
        flipped = False
        if c_l is not None and c_r is not None:
            c_r, flipped = align_orientation(c_l, c_r)

        # annotated side-by-side image for inspection
        for img, c, how, side in ((img_l, c_l, how_l, "left"), (img_r, c_r, how_r, "right")):
            if c is not None:
                cv2.drawChessboardCorners(img, PATTERN, c, True)
            text = f"{side}: {how}" + (" (order flipped)" if flipped and side == "right" else "")
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 0) if c is not None else (0, 0, 255), 2)
        cv2.imwrite(f"{DEBUG_DIR}/detections/{name}", np.hstack([img_l, img_r]))

        print(f"  {name}: left {how_l:<12} right {how_r:<12}" + ("  (right order flipped)" if flipped else ""))
        views.append(dict(name=name, left_file=lf, right_file=rf, left=c_l, right=c_r, flipped=flipped))

    n_l = sum(v["left"] is not None for v in views)
    n_r = sum(v["right"] is not None for v in views)
    n_pairs = sum(v["left"] is not None and v["right"] is not None for v in views)
    print(f"  -> image size {image_size}; found: left {n_l}, right {n_r}, both {n_pairs} (of {len(views)}); "
          f"{sum(v['flipped'] for v in views)} pairs needed corner-order correction")
    print(f"  -> annotated images in {DEBUG_DIR}/detections/")
    return views, image_size


# ----------------------------------------------------------------------------
# STEP 2: SINGLE-CAMERA CALIBRATION
# ----------------------------------------------------------------------------

def calibrate_camera(image_points, image_size):
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [OBJP] * len(image_points), image_points, image_size, None, None, flags=INTRINSIC_FLAGS)
    return camera_matrix, distortion, rms


# ----------------------------------------------------------------------------
# STEP 3 + 4: STEREO CALIBRATION, RECTIFICATION
# ----------------------------------------------------------------------------
def calibrate_stereo(pairs, K1, D1, K2, D2, image_size):
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        [OBJP] * len(pairs), [p["left"] for p in pairs], [p["right"] for p in pairs],
        K1, D1, K2, D2, image_size, criteria=STEREO_CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"  {len(pairs)} pairs, RMS {rms:.3f} px")
    print(f"  baseline = {np.linalg.norm(T):.1f} mm (compare with a tape measure), "
          f"T = {T.ravel().round(1)} mm (x should dominate)")
    print(f"  rotation [deg] = {np.degrees(cv2.Rodrigues(R)[0]).ravel().round(2)}")
    return R, T, E, F


def epipolar_error(pairs, cal):
    """After rectification matching corners must lie on the same row."""
    dy = []
    for p in pairs:
        y_l = cv2.undistortPoints(p["left"], cal["K1"], cal["D1"], R=cal["R1"], P=cal["P1"]).reshape(-1, 2)[:, 1]
        y_r = cv2.undistortPoints(p["right"], cal["K2"], cal["D2"], R=cal["R2"], P=cal["P2"]).reshape(-1, 2)[:, 1]
        dy.append(np.abs(y_l - y_r))
    dy = np.concatenate(dy)
    print(f"  vertical misalignment: mean {dy.mean():.2f} px, max {dy.max():.2f} px (want mean < ~0.5)")


# ----------------------------------------------------------------------------
# STAGE PREVIEWS
# ----------------------------------------------------------------------------
def to_int(p):
    return tuple(int(round(v)) for v in p)


def epipolar_curve(E, left_normalised, K_right, D_right):
    """Epipolar line in the right camera for one left point, projected into the right image.
    With lens distortion (D_right != 0) it is a curve, hence the sampling."""
    a, b, c = E @ np.array([*left_normalised, 1.0])      # a*x + b*y + c = 0 in normalised coords
    xs = np.linspace(-1.5, 1.5, 120)
    points = np.stack([xs, -(a * xs + c) / b, np.ones_like(xs)], axis=1)
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K_right, D_right)
    return np.round(pixels.reshape(-1, 2)).astype(np.int32)


def warp(img, pts, K, D, R, P, size):
    """Undistort (+ rectify if R is not identity) an image and the corner points with it."""
    maps = cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
    return cv2.remap(img, *maps, cv2.INTER_LINEAR), cv2.undistortPoints(pts, K, D, R=R, P=P).reshape(-1, 2)


def draw_stage(img_l, img_r, pts_l, pts_r, lines, lines_on_both, path):
    """Circle the picked corners in both images; draw the epipolar line of each LEFT corner.
    The circle in the right image should sit on the line of the same colour."""
    img_l, img_r = img_l.copy(), img_r.copy()
    for n, (pl, pr, line) in enumerate(zip(pts_l, pts_r, lines)):
        color = COLORS[n % len(COLORS)]
        cv2.polylines(img_r, [line], False, color, 1, cv2.LINE_AA)
        if lines_on_both:
            cv2.polylines(img_l, [line], False, color, 1, cv2.LINE_AA)
        cv2.circle(img_l, to_int(pl), 5, color, 1, cv2.LINE_AA)
        cv2.circle(img_r, to_int(pr), 5, color, 1, cv2.LINE_AA)
    cv2.imwrite(path, np.hstack([img_l, img_r]))


def save_stage_previews(pair, cal):
    """Four images: original -> undistorted -> rectified (all pixels, black borders kept) -> final rectified."""
    K1, D1, K2, D2, E, size = (cal[k] for k in ("K1", "D1", "K2", "D2", "E", "image_size"))
    w = size[0]
    img_l, img_r = cv2.imread(pair["left_file"]), cv2.imread(pair["right_file"])
    picks = list(range(0, len(OBJP), 5))                                  # every 5th corner -> 7 lines
    left, right = pair["left"], pair["right"]
    normalised = cv2.undistortPoints(left, K1, D1).reshape(-1, 2)[picks]  # left corners as (X/Z, Y/Z)
    stem = os.path.splitext(pair["name"])[0]
    out = lambda stage: f"{DEBUG_DIR}/stages/{stem}_{stage}.png"

    # 1. original: epipolar curves in the distorted right image
    curves = [epipolar_curve(E, n, K2, D2) for n in normalised]
    draw_stage(img_l, img_r, left.reshape(-1, 2)[picks], right.reshape(-1, 2)[picks], curves, False,
               out("1_original"))

    # 2. undistorted, keeping every pixel (alpha=1 -> black borders): epipolar lines are now straight
    I = np.eye(3)
    newK1 = cv2.getOptimalNewCameraMatrix(K1, D1, size, 1)[0]
    newK2 = cv2.getOptimalNewCameraMatrix(K2, D2, size, 1)[0]
    il, pl = warp(img_l, left, K1, D1, I, newK1, size)
    ir, pr = warp(img_r, right, K2, D2, I, newK2, size)
    curves = [epipolar_curve(E, n, newK2, np.zeros(5)) for n in normalised]
    draw_stage(il, ir, pl[picks], pr[picks], curves, False, out("2_undistorted"))

    # 3. rectified keeping every pixel, 4. rectified with the final ALPHA: lines are horizontal rows
    full = cv2.stereoRectify(K1, D1, K2, D2, size, cal["R"], cal["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    for stage, (R1, R2, P1, P2) in (("3_rectified_all_pixels", full[:4]),
                                    ("4_rectified_final", (cal["R1"], cal["R2"], cal["P1"], cal["P2"]))):
        il, pl = warp(img_l, left, K1, D1, R1, P1, size)
        ir, pr = warp(img_r, right, K2, D2, R2, P2, size)
        rows = [np.array([[0, int(round(y))], [w, int(round(y))]], np.int32) for y in pl[picks, 1]]
        draw_stage(il, ir, pl[picks], pr[picks], rows, True, out(stage))


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    section("GLOSSARY")
    for term, meaning in GLOSSARY.items():
        print(f"  {term:<22} {meaning}")

    section("1. Corner detection")
    views, image_size = detect_all()
    for v in views:
        if v["name"] == "024.png":
            v["right"] = None

    section("2. Per-camera calibration (all boards found by that camera)")
    K, D = {}, {}
    for side in ("left", "right"):
        used_views = [view for view in views if view[side] is not None]
        K[side], D[side], rms = calibrate_camera([view[side] for view in used_views], image_size)
        print(f"  [{side}] {len(used_views)} views, RMS {rms:.3f} px")

    section("3. Stereo calibration (pairs found in both cameras, intrinsics fixed)")
    pairs = [view for view in views if view["left"] is not None and view["right"] is not None]
    if not pairs:
        raise RuntimeError("No usable image pair")
    R, T, E, F = calibrate_stereo(pairs, K["left"], D["left"], K["right"], D["right"], image_size)

    section("4. Rectification")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K["left"], D["left"], K["right"], D["right"], image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=ALPHA)
    print(f"  alpha = {ALPHA}: valid ROI left {roi1}, right {roi2} (image is {image_size[0]}x{image_size[1]})")
    cal = dict(image_size=image_size, K1=K["left"], D1=D["left"], K2=K["right"], D2=D["right"],
               R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2)
    epipolar_error(pairs, cal)

    section("5. Saving")
    np.savez(OUTPUT_FILE, **cal)
    print(f"  calibration saved to {OUTPUT_FILE}")
    os.makedirs(f"{DEBUG_DIR}/stages", exist_ok=True)
    for pair in pairs[::max(1, len(pairs) // PREVIEW_PAIRS)][:PREVIEW_PAIRS]:
        save_stage_previews(pair, cal)
    print(f"  stage previews saved to {DEBUG_DIR}/stages/")


if __name__ == "__main__":
    main()
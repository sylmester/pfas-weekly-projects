import glob

import cv2
import numpy as np

LEFT_GLOB = "0_calib/left/*.png"
RIGHT_GLOB = "0_calib/right/*.png"
PATTERN = (7, 5)  # inner corners (columns, rows)
SQUARE_SIZE_MM = 80.0
SCALES = (1.0, 0.5, 0.25, 1.25, 1.5, 2.0, 2.5)
EXCLUDED_RIGHT_IMAGE = "024.png"  # corners misplaced, found by visual inspection

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Known 3D corner positions on the board in mm, shape (35, 3)
board_points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
board_points[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM


def refine_corners(gray_image, corners):
    # corners: (35, 1, 2)
    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
    horizontal_gaps = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical_gaps = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    square_size_px = min(np.median(horizontal_gaps), np.median(vertical_gaps))

    half_window = int(0.45 * square_size_px)
    refined = cv2.cornerSubPix(gray_image, corners, (half_window, half_window), (-1, -1), SUBPIX_CRITERIA)
    return refined.reshape(-1, 2)  # (35, 2)


def find_corners(gray_image):
    sector_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # Sector-based detector on all scales first, then the classic detector
    for detector_name in ("sector", "classic"):
        for scale in SCALES:
            if scale < 1:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC
            scaled_image = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=interpolation)

            if detector_name == "sector":
                found, corners = cv2.findChessboardCornersSB(scaled_image, PATTERN, sector_flags)
            else:
                found, corners = cv2.findChessboardCorners(scaled_image, PATTERN, classic_flags)

            if found:
                corners_full_size = (corners / scale).astype(np.float32)
                return refine_corners(gray_image, corners_full_size)
    return None


def match_corner_order(left_corners, right_corners):
    # The board looks the same rotated 180 deg, so reverse the right order if it points the other way
    left_direction = left_corners[-1] - left_corners[0]    # (2,)
    right_direction = right_corners[-1] - right_corners[0]  # (2,)
    if np.dot(left_direction, right_direction) < 0:
        return right_corners[::-1].copy()
    return right_corners


# 1. Corner detection
left_files = sorted(glob.glob(LEFT_GLOB))
right_files = sorted(glob.glob(RIGHT_GLOB))

left_detections = []
right_detections = []
flipped_count = 0

for left_file, right_file in zip(left_files, right_files):
    left_image = cv2.cvtColor(cv2.imread(left_file), cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(cv2.imread(right_file), cv2.COLOR_BGR2GRAY)
    image_size = left_image.shape[::-1]  # (width, height)

    left_corners = find_corners(left_image)
    right_corners = find_corners(right_image)

    

    if right_file.endswith(EXCLUDED_RIGHT_IMAGE):
        right_corners = None
    if left_corners is not None and right_corners is not None:
        ordered_corners = match_corner_order(left_corners, right_corners)
        if ordered_corners is not right_corners:
            flipped_count += 1
        right_corners = ordered_corners

    left_detections.append(left_corners)
    right_detections.append(right_corners)

# 2. Intrinsics per camera, on all of its own detections
intrinsic_flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

left_used = [corners for corners in left_detections if corners is not None]
right_used = [corners for corners in right_detections if corners is not None]

left_rms, left_matrix, left_distortion, _, _ = cv2.calibrateCamera(
    [board_points] * len(left_used), left_used, image_size, None, None, flags=intrinsic_flags)
right_rms, right_matrix, right_distortion, _, _ = cv2.calibrateCamera(
    [board_points] * len(right_used), right_used, image_size, None, None, flags=intrinsic_flags)

# 3. Stereo extrinsics on the pairs found in both cameras, intrinsics fixed
left_pairs = []
right_pairs = []
for left_corners, right_corners in zip(left_detections, right_detections):
    if left_corners is not None and right_corners is not None:
        left_pairs.append(left_corners)
        right_pairs.append(right_corners)

stereo_rms, _, _, _, _, rotation, translation, essential, _ = cv2.stereoCalibrate(
    [board_points] * len(left_pairs), left_pairs, right_pairs,
    left_matrix, left_distortion, right_matrix, right_distortion,
    image_size, criteria=STEREO_CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)

# 4. Rectification
left_rotation, right_rotation, left_projection, right_projection, disparity_to_depth, _, _ = cv2.stereoRectify(
    left_matrix, left_distortion, right_matrix, right_distortion, image_size,
    rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

# Vertical misalignment of matching corners after rectification
vertical_offsets = []
for left_corners, right_corners in zip(left_pairs, right_pairs):
    left_rectified = cv2.undistortPoints(left_corners, left_matrix, left_distortion,
                                         R=left_rotation, P=left_projection).reshape(-1, 2)    # (35, 2)
    right_rectified = cv2.undistortPoints(right_corners, right_matrix, right_distortion,
                                          R=right_rotation, P=right_projection).reshape(-1, 2)  # (35, 2)
    vertical_offsets.append(np.abs(left_rectified[:, 1] - right_rectified[:, 1]))
vertical_offsets = np.concatenate(vertical_offsets)

print(f"left: {len(left_used)} views, RMS {left_rms:.3f} px")
print(f"right: {len(right_used)} views, RMS {right_rms:.3f} px")
print(f"stereo: {len(left_pairs)} pairs, RMS {stereo_rms:.3f} px")
print(f"baseline {np.linalg.norm(translation):.1f} mm, T = {translation.ravel().round(1)} mm")
print(f"rotation [deg] = {np.degrees(cv2.Rodrigues(rotation)[0]).ravel().round(2)}")
print(f"vertical misalignment: mean {vertical_offsets.mean():.2f} px, max {vertical_offsets.max():.2f} px")
print(f"images: {len(left_files)}, corners found left {len(left_used)}, right {len(right_used)}")
print(f"flipped pairs: {flipped_count}")
print(f"left: fx, fy = {left_matrix[0, 0]:.1f}, {left_matrix[1, 1]:.1f}, cx, cy = {left_matrix[0, 2]:.1f}, {left_matrix[1, 2]:.1f}")
print(f"right: fx, fy = {right_matrix[0, 0]:.1f}, {right_matrix[1, 1]:.1f}, cx, cy = {right_matrix[0, 2]:.1f}, {right_matrix[1, 2]:.1f}")
print(f"left distortion (k1, k2, p1, p2, k3) = {left_distortion.ravel().round(4)}")
print(f"right distortion (k1, k2, p1, p2, k3) = {right_distortion.ravel().round(4)}")

np.savez("calibration.npz", image_size=image_size,
         K1=left_matrix, D1=left_distortion, K2=right_matrix, D2=right_distortion,
         R=rotation, T=translation, E=essential,
         R1=left_rotation, R2=right_rotation, P1=left_projection, P2=right_projection,
         Q=disparity_to_depth)
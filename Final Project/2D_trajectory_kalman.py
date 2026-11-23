from pathlib import Path
from collections import deque
import re
import cv2
import numpy as np

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DIR = BASE_DIR / '4_circle'
INPUT_DIR = DIR / 'rectified_left'
OUTPUT_DIR = DIR / '2D_trajectory_kalman'
FPS = 30.0
TRAJECTORY_HEIGHT = 300
DIFF_THRESHOLD = 15
DIFF_FRAME_GAP = 3
SHOW_MOTION_MASK = True
MIN_AREA, MAX_AREA = 200, 50000
MIN_WIDTH_PX, MAX_WIDTH_PX = 10, 300
MIN_HEIGHT_PX, MAX_HEIGHT_PX = 10, 300
if DIR.name == '2_obstacle':
    MIN_AREA, MAX_AREA = 7000, 50000
    MIN_WIDTH_PX, MAX_WIDTH_PX = 50, 300
    MIN_HEIGHT_PX, MAX_HEIGHT_PX = 50, 300

def update(x, P, Z, H, R):
    y = Z - np.dot(H, x)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    x = x + np.dot(K, y)
    P = np.dot((np.eye(6) - np.dot(K, H)), P)
    return [x, P]


def predict(x, P, F, u):
    x = np.dot(F, x) + u
    P = np.dot(F, np.dot(P, F.T))
    return [x, P]


def detect_robot(previous, current):
    difference = cv2.absdiff(previous, current)
    _, mask = cv2.threshold(difference, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    # Join nearby regions before removing noise.
    # Closing: dilation followed by erosion.
    mask = cv2.dilate(mask, np.ones((9, 31), np.uint8))
    mask = cv2.erode(mask, np.ones((5, 31), np.uint8))
    # Opening: erosion followed by dilation.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if SHOW_MOTION_MASK:
        cv2.imshow('Motion mask', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (MIN_AREA <= area <= MAX_AREA
                and MIN_WIDTH_PX <= w <= MAX_WIDTH_PX
                and MIN_HEIGHT_PX <= h <= MAX_HEIGHT_PX):
            center = np.array([[x + w / 2], [y + h / 2]])
            detections.append((center, (w, h), area))
    return detections


def draw_trajectories(width, height, measured, predicted):
    panel = np.zeros((TRAJECTORY_HEIGHT, width, 3), dtype=np.uint8)
    scale_x = (width - 30) / max(1, width - 1)
    scale_y = (TRAJECTORY_HEIGHT - 60) / max(1, height - 1)

    def to_panel(point):
        return (int(round(15 + point[0] * scale_x)),
                int(round(45 + point[1] * scale_y)))

    for points, color, thickness in ((predicted, (0, 165, 255), 5),
                                     (measured, (0, 255, 0), 2)):
        for start, end in zip(points, points[1:]):
            if start is not None and end is not None:
                cv2.line(panel, to_panel(start), to_panel(end), color, thickness,
                         cv2.LINE_AA)
        if points and points[-1] is not None:
            cv2.circle(panel, to_panel(points[-1]), 4, color, -1)
    cv2.putText(panel, 'Detected: green | Next-frame prediction: orange',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def main():
    paths = sorted(INPUT_DIR.glob('*.png'), key=lambda p: [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r'(\d+)', p.name)])
    if len(paths) <= DIFF_FRAME_GAP:
        raise ValueError(f'At least {DIFF_FRAME_GAP + 1} images are required in {INPUT_DIR}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dt = 1 / FPS
    # State: [x, y, vx, vy, ax, ay].
    F = np.array([[1, 0, dt, 0, dt**2/2, 0],
                  [0, 1, 0, dt, 0, dt**2/2],
                  [0, 0, 1, 0, dt, 0],
                  [0, 0, 0, 1, 0, dt],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]], dtype=float)
    F_occlusion = np.eye(6)
    F_occlusion[0, 2] = F_occlusion[1, 3] = dt
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]], dtype=float)
    R = np.eye(2) * 100
    u = np.zeros((6, 1))
    state = None
    prediction_anchor = None
    gray_history = deque(maxlen=DIFF_FRAME_GAP)
    writer = None
    missed = 0
    frame_size = None
    measured_trajectory = []
    predicted_trajectory = []

    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Cannot read {path}')
            height, width = frame.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(str(OUTPUT_DIR / 'result.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                         (width, height + TRAJECTORY_HEIGHT))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create the video')
                # Resize the full image and its panel to fit on the screen.
                # The video is saved at its original resolution.
                cv2.namedWindow('Motion and Kalman', cv2.WINDOW_NORMAL)
                display_scale = min(1.0, 1100 / width,
                                    750 / (height + TRAJECTORY_HEIGHT))
                cv2.resizeWindow('Motion and Kalman', round(width * display_scale),
                                 round((height + TRAJECTORY_HEIGHT) * display_scale))
                if SHOW_MOTION_MASK:
                    cv2.namedWindow('Motion mask', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Motion mask', round(width * display_scale),
                                     round(height * display_scale))
            if (width, height) != frame_size:
                raise ValueError('All images must have the same size')
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            detections = (detect_robot(gray_history[0], gray)
                          if len(gray_history) == DIFF_FRAME_GAP else [])
            detection = max(detections, key=lambda d: d[2]) if detections else None
            predicted_point = None

            if state is None:
                if detection is not None:
                    state = np.zeros((6, 1))
                    state[:2] = detection[0]
                    P = np.diag([100., 100., 10000., 10000., 10000., 10000.])
            else:
                old_state, old_P = state.copy(), P.copy()
                state[4:6] = 0
                state, P = predict(state, P, F_occlusion, u)
                if detection is not None:
                    if missed == 0:
                        state, P = predict(old_state, old_P, F, u)
                    state, P = update(state, P, detection[0], H, R)

            if detection is not None:
                # Store the latest detection with the velocity estimated by Kalman.
                prediction_anchor = state.copy()
                prediction_anchor[:2] = detection[0]
                cx, cy = detection[0][:, 0]
                w, h = detection[1]
                x0, y0 = int(cx - w/2 - 10), int(cy - h/2 - 10)
                x1, y1 = int(cx + w/2 + 10), int(cy + h/2 + 10)
                if x1 >= 0 and y1 >= 0 and x0 < width and y0 < height:
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(width-1, x1), min(height-1, y1)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(frame, 'Detected', (x0, max(20, y0-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if state is not None:
                missed = missed + 1 if detection is None else 0
                # Always calculate from the latest detection without chaining
                # the drawn points. Without a detection, increase the elapsed time.
                next_F = F.copy() if detection is not None else F_occlusion.copy()
                if detection is None:
                    next_F[0, 2] = next_F[1, 3] = (missed + 1) * dt
                forecast = prediction_anchor.copy()
                if detection is None:
                    # Keep vx, vy from the latest detection, without acceleration.
                    forecast[4:6] = 0
                next_state, _ = predict(forecast, P.copy(), next_F, u)
                predicted_point = tuple(float(v) for v in next_state[:2, 0])
                px, py = (int(round(v)) for v in predicted_point)
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(frame, (px, py), 5, (0, 165, 255), -1)

            # Green trajectory: actual measurements; interrupted when there is no detection.
            measured_point = (tuple(float(v) for v in detection[0][:, 0])
                              if detection is not None else None)
            measured_trajectory.append(measured_point)
            predicted_trajectory.append(predicted_point)
            panel = draw_trajectories(width, height, measured_trajectory,
                                      predicted_trajectory)
            display = np.vstack((frame, panel))
            gray_history.append(gray)
            writer.write(display)
            cv2.imshow('Motion and Kalman', display)
            if cv2.waitKey(max(1, round(1000 / FPS))) & 0xFF in (27, ord('q')):
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

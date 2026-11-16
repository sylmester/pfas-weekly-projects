from pathlib import Path
import re
import cv2
import numpy as np

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DIR = BASE_DIR / '2_obstacle'
INPUT_DIR = DIR / 'rectified_left'
OUTPUT_DIR = DIR / 'motion_temporal_kalman_last'
FPS = 30.0
DIFF_THRESHOLD = 15
MIN_AREA, MAX_AREA = 800, 50000
MIN_WIDTH_PX, MAX_WIDTH_PX = 25, 300
MIN_HEIGHT_PX, MAX_HEIGHT_PX = 25, 300


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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
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


def main():
    # Natural order: 1.png, 2.png, ..., 10.png.
    paths = sorted(INPUT_DIR.glob('*.png'), key=lambda p: [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r'(\d+)', p.name)])
    if len(paths) < 2:
        raise ValueError(f'At least two images are required in {INPUT_DIR}')
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
    previous = None
    writer = None
    missed = 0
    frame_size = None

    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Cannot read {path}')
            height, width = frame.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(str(OUTPUT_DIR / 'result.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), FPS, frame_size)
                if not writer.isOpened():
                    raise RuntimeError('Cannot create the video')
            if (width, height) != frame_size:
                raise ValueError('All images must have the same size')
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            detections = detect_robot(previous, gray) if previous is not None else []
            detection = None

            if state is None:
                if detections:
                    detection = max(detections, key=lambda d: d[2])
                    state = np.zeros((6, 1))
                    state[:2] = detection[0]
                    P = np.diag([100., 100., 10000., 10000., 10000., 10000.])
                    box_size = np.array(detection[1], dtype=float)
            else:
                old_state, old_P = state.copy(), P.copy()
                # Search near a constant-velocity prediction.
                state[4:6] = 0
                state, P = predict(state, P, F_occlusion, u)
                if detections:
                    candidate = min(detections, key=lambda d: np.linalg.norm(d[0] - state[:2]))
                    if np.linalg.norm(candidate[0] - state[:2]) <= 150:
                        detection = candidate
                if detection is not None:
                    if missed == 0:
                        state, P = predict(old_state, old_P, F, u)
                    state, P = update(state, P, detection[0], H, R)
                    box_size = np.array(detection[1], dtype=float)

            if state is not None:
                missed = missed + 1 if detection is None else 0
                color = (0, 165, 255) if missed else (0, 255, 0)
                label = 'Predicted' if missed else 'Detected'
                cx, cy = state[:2, 0]
                w, h = box_size
                x0, y0 = int(cx - w/2 - 10), int(cy - h/2 - 10)
                x1, y1 = int(cx + w/2 + 10), int(cy + h/2 + 10)
                if x1 >= 0 and y1 >= 0 and x0 < width and y0 < height:
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(width-1, x1), min(height-1, y1)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, label, (x0, max(20, y0-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            previous = gray
            writer.write(frame)
            cv2.imshow('Motion and Kalman', frame)
            if cv2.waitKey(max(1, round(1000 / FPS))) & 0xFF in (27, ord('q')):
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

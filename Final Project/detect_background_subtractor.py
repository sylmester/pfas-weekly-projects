import copy
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import DIR, RECTIFIED_LEFT_GLOB

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
OUTPUT_DIR = f"{DIR}/subtract_background/"
image_paths = sorted(glob.glob(RECTIFIED_LEFT_GLOB))

MIN_AREA = 250    # ignore small blobs (noise, shadows)
MAX_AREA = 50000  # ignore huge blobs (lighting changes)
MAX_WIDTH = 250
MAX_HEIGHT = 250

frame_idx = 0

# Set up background subtractor Mixture of gaussian MOG2 (We can also try out GMG later)
backsub = cv2.createBackgroundSubtractorMOG2(
    varThreshold=30,    # higher = stricter, fewer false positives
    detectShadows=False # shadows are gray, would pollute the mask
)

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def detect_husky(frame):
    global frame_idx
    lr = -1 if frame_idx < 75 else 0.002 # Set the leraning rate higher in the start
    frame_idx += 1
    mask = backsub.apply(frame, learningRate=lr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Biggest moving blob = assumed Husky
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = w*h

    if area < MIN_AREA or area > MAX_AREA or w > MAX_WIDTH or h > MAX_HEIGHT:
        return None
    else:
        return x, y, w, h, area, mask

def draw_detection(frame, det):
    x, y, w, h, area, mask = det
    frame[mask > 0] = (0.5 * frame[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8) # Blend each bgr with red
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    labels = [f"Husky {int(area)} px", f"W {w} px", f"H {h} px"]
    for i, t in enumerate(labels):
        cv2.putText(frame, t, (x, max(15, y - 10 - 20 * i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        det = detect_husky(frame)
        if det:
            frame = draw_detection(frame, det)

        cv2.imshow("detection", frame)

        name = os.path.basename(path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()









# ----------------------------------------------------------------------------
# Old code
# ----------------------------------------------------------------------------
# The simlple background subtract function worked as well and even better than more complicated attempts

# def detect_husky(frame, morphology=True):
#     global frame_idx
#     lr = -1 if frame_idx < 75 else 0.002
#     frame_idx += 1
#     mask = backsub.apply(frame, learningRate=lr)

#     if morphology:
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                    cv2.CHAIN_APPROX_SIMPLE)
#     boxes = [cv2.boundingRect(c) for c in contours]
#     keep = [(c, b) for c, b in zip(contours, boxes)
#             if cv2.contourArea(c) > 30 and b[2] < MAX_WIDTH and b[3] < MAX_HEIGHT]
#     if not keep:
#         return None

#     keep.sort(key=lambda cb: cv2.contourArea(cb[0]), reverse=True)
#     x0, y0, w0, h0 = keep[0][1]
#     c0 = np.array([x0 + w0 / 2, y0 + h0 / 2])

#     R = 80
#     keep = [(c, b) for c, b in keep
#             if np.linalg.norm(np.array([b[0] + b[2] / 2, b[1] + b[3] / 2]) - c0) < R]

#     x, y, w, h = cv2.boundingRect(np.vstack([c for c, _ in keep]))
#     area = w * h
#     if area < MIN_AREA or area > MAX_AREA:
#         return None
#     return x, y, w, h, area, mask
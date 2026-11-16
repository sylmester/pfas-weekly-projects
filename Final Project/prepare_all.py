"""
Rectify + compute disparity for ALL sequences with one command, by running
Sylvester's scripts once per sequence:

    python rectify.py <seq>        -> <seq>/rectified_left, <seq>/rectified_right
    python disparity_map.py <seq>  -> <seq>/disparity, <seq>/depth

Re-run it whenever calibration.npz changes, so every sequence uses the same calibration.
"""
import os
import subprocess
import sys

SEQUENCES = ["1_no_obstacle", "2_obstacle", "3_curved_bend", "4_circle"]
HERE = os.path.dirname(os.path.abspath(__file__))  # the scripts look for the data next to them

for seq in SEQUENCES:
    print(f"===== {seq} =====", flush=True)
    subprocess.run([sys.executable, "rectify.py", seq], cwd=HERE, check=True)
    subprocess.run([sys.executable, "disparity_map.py", seq], cwd=HERE, check=True)

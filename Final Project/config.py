DIR = "3_curved_bend"  # 0_calib, 1_no_obstacle, 2_obstacle, 3_curved_bend

DISP_DIR = f"{DIR}/disparity/"
DEPTH_DIR = f"{DIR}/depth/"
DEPTH_GLOB = f"{DIR}/depth/*.npy"
LEFT_GLOB = f"{DIR}/rectified_left/*.png"
RIGHT_GLOB = f"{DIR}/rectified_right/*.png"

CALIB_FILE = "calibration.npz"


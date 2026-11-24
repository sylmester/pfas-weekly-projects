DIR = "3_curved_bend"  # 0_calib, 1_no_obstacle, 2_obstacle, 3_curved_bend

LEFT_GLOB = f"{DIR}/left/*.png"
RIGHT_GLOB = f"{DIR}/right/*.png"


DISP_DIR = f"{DIR}/disparity/"
DEPTH_DIR = f"{DIR}/depth/"


RECTIFIED_LEFT_GLOB = f"{DIR}/rectified_left/*.png"
RECTIFIED_RIGHT_GLOB = f"{DIR}/rectified_right/*.png"
DEPTH_GLOB = f"{DIR}/depth/*.npy"


CALIB_FILE = "calibration.npz"


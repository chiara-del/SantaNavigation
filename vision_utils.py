import cv2
import numpy as np
import math
import sys

# --- ARUCO INITIALIZATION ---
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
except AttributeError:
    print("Error: ArUco dictionary not found. Is 'opencv-python-contrib' installed?")
    sys.exit()

# --- CAMERA & MATRIX UTILS ---

def setup_camera(camera_index, width, height):
    """Initializes and configures the webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {camera_index}.")
        return None
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Webcam {camera_index} opened successfully.")
    return cap

def load_transform_matrix(path):
    """Loads the perspective transform matrix from a .npy file."""
    try:
        matrix = np.load(path)
        print(f"Loaded perspective transform matrix from {path}.")
        return matrix
    except FileNotFoundError:
        print(f"Info: Matrix file not found at {path}.")
        return None

def run_calibration_wizard(cap, map_width, map_height, matrix_file_path):
    """
    Guides the user through the (Step 1) perspective calibration.
    Takes a snapshot, gets 4 clicks, calculates, saves, and returns the matrix.
    """
    
    # These variables will be shared with the inner callback function
    src_points = []
    calibration_frame = None

    def _click_event_callback(event, x, y, flags, params):
        """Inner function to handle mouse clicks during calibration."""
        nonlocal src_points, calibration_frame
        
        if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
            src_points.append((x, y))
            
            # Draw feedback on the image
            cv2.circle(calibration_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibrate - Click 4 Corners", calibration_frame)
            print(f"Point {len(src_points)} added: ({x}, {y})")
            
            if len(src_points) == 4:
                print("All 4 points selected. Press 'c' to save and continue.")

    # 1. Get snapshot from webcam
    print("\n--- Calibration Wizard ---")
    print("Press 's' in the preview window to capture a frame for calibration.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            return None
        
        preview = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Webcam Preview - Press 's' to snapshot", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            calibration_frame = frame.copy()
            cv2.destroyWindow("Webcam Preview - Press 's' to snapshot")
            break

    # 2. Select 4 corners
    cv2.namedWindow("Calibrate - Click 4 Corners")
    cv2.setMouseCallback("Calibrate - Click 4 Corners", _click_event_callback)
    
    print("\nClick on the 4 corners of your arena in this order:")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Bottom-Right")
    print("  4. Bottom-Left")
    print("After 4 clicks, press 'c' to continue.")
    
    cv2.imshow("Calibrate - Click 4 Corners", calibration_frame)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if len(src_points) == 4:
                break
            else:
                print("Please click exactly 4 points before pressing 'c'.")

    cv2.destroyWindow("Calibrate - Click 4 Corners")

    # 3. Calculate and save matrix
    print("Calculating perspective transform matrix...")
    src_points_np = np.float32(src_points)
    dst_points_np = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points_np, dst_points_np)
    np.save(matrix_file_path, matrix)
    print(f"Matrix calculated and saved successfully to {matrix_file_path}!")
    
    return matrix


# --- DETECTION & DRAWING UTILS (Step 2) ---

def detect_aruco_markers(frame):
    """
    Detects ALL ArUco markers in a frame.
    Returns: A dictionary {id: ((x, y), angle_deg), ...}
    """
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
    
    poses = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            bottom_mid = (marker_corners[2] + marker_corners[3]) / 2
            top_mid = (marker_corners[0] + marker_corners[1]) / 2
            
            dx = top_mid[0] - bottom_mid[0]
            dy = top_mid[1] - bottom_mid[1]
            
            angle_rad = math.atan2(dx, -dy) 
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
                
            poses[marker_id] = ((center_x, center_y), angle_deg)
    
    return poses

def detect_obstacles_hsv(frame, lower_hsv, upper_hsv, min_area):
    """
    Detects obstacle contours using HSV color segmentation.
    Returns: (list_of_valid_contours, binary_mask_for_debugging)
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, mask_cleaned


def detect_obstacles_grayscale(frame, threshold_value, min_area):
    """
    Detects bright obstacles using grayscale thresholding.
    Returns: (list_of_valid_contours, binary_mask_for_debugging)
    """
    # 1. Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply binary threshold
    #    Pixels > threshold_value become 255 (white), others 0 (black)
    _, mask = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 3. Find contours on the mask
    #    (No cleanup/morphology needed, but you could add it)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Filter contours by area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, mask


def draw_all_detections(frame, thymio_pose, goal_pos, obstacles, thymio_id, goal_id):
    """
    Draws all detected elements onto the display frame.
    Modifies the frame in-place.
    """
    if thymio_pose is not None:
        pos, angle = thymio_pose
        cv2.circle(frame, pos, 10, (0, 255, 255), -1) # Yellow
        angle_rad = math.radians(angle)
        end_x = int(pos[0] + 30 * math.sin(angle_rad))
        end_y = int(pos[1] - 30 * math.cos(angle_rad))
        cv2.arrowedLine(frame, pos, (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, f"Thymio (ID {thymio_id})", 
                    (pos[0] + 15, pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if goal_pos is not None:
        cv2.circle(frame, goal_pos, 15, (255, 0, 0), -1) # Blue
        cv2.putText(frame, f"Goal (ID {goal_id})", (goal_pos[0] + 15, goal_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.drawContours(frame, obstacles, -1, (0, 0, 255), 2) # Red outlines
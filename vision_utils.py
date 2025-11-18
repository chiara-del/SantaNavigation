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
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
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
    Guides the user through perspective calibration.
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

def mask_to_polygons(
    mask,
    blur_ksize=5,
    morph_kernel_size=5,
    morph_iters=2,
    epsilon_factor=0.02,
    min_area=100
):
    """
    Convert a binary mask (white shapes on black) into simplified polygons.

    Parameters
    ----------
    mask : np.ndarray
        Binary image (uint8). Shapes should be white (255), background black (0).
        Can also be grayscale; we'll threshold it.
    blur_ksize : int
        Kernel size for Gaussian blur (must be odd). 0 or 1 to disable.
    morph_kernel_size : int
        Kernel size for morphological operations (closing + opening).
    morph_iters : int
        Number of iterations for morphology.
    epsilon_factor : float
        Simplification factor for approxPolyDP. Higher = fewer vertices.
        Typical range: 0.01–0.05.
    min_area : float
        Minimum contour area to keep (to filter tiny noise).

    Returns
    -------
    polygons : list of np.ndarray
        Each element is an array of shape (N, 2) with (x, y) vertex coordinates.
        Example: [array([[x1, y1], [x2, y2], ...]), ...]
    """

    # 1) Ensure single channel
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    # 2) Optional blur to smooth noise before threshold
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        mask_gray = cv2.GaussianBlur(mask_gray, (blur_ksize, blur_ksize), 0)

    # 3) Binarize (in case it's not strictly 0/255 already)
    _, th = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4) Morphological operations to fill gaps and remove specks
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 5) Find contours on cleaned mask
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # ignore tiny blobs

        # Optional: uncomment next line if you want convex obstacles only
        # cnt = cv2.convexHull(cnt)

        # 6) Simplify contour -> polygon
        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # shape: (N, 1, 2)

        # Reshape to (N, 2) and store as int
        poly = approx.reshape(-1, 2)
        polygons.append(poly)

    return polygons

def detect_obstacles_hsv(frame, lower_hsv, upper_hsv, min_area, robot_radius):
    """
    Detects obstacles using HSV, cleans noise, and expands them by robot_radius.
    Returns: (list_of_expanded_contours, c_space_mask)
    """
    # 1. Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Threshold
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    # 3. Noise Removal (The "Cleaning" Step)
    # We use a small fixed kernel (5x5) to remove tiny speckles
    # If we didn't do this, tiny noise dots would become giant circles in step 4!
    clean_kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    
    # 4. Expansion (C-Space Generation)
    # We dilate the cleaned mask by the robot radius.
    # The kernel size (diameter) must be radius * 2
    dilation_diameter = int(robot_radius * 2)
    
    # Use an ELLIPSE kernel for smooth, circular expansion
    if dilation_diameter > 0:
        expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_diameter, dilation_diameter))
        mask_expanded = cv2.dilate(mask_cleaned, expansion_kernel)
    else:
        mask_expanded = mask_cleaned # Should not happen if radius > 0
    
    # 5. Find Contours (Vertices) on the EXPANDED mask
    #contours, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filter by Area
    #valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, mask_expanded


def detect_obstacles_grayscale(frame, threshold_value, min_area, robot_radius):
    """
    Detects obstacles using Grayscale, cleans noise, and expands them by robot_radius.
    Returns: (list_of_expanded_contours, c_space_mask)
    """
    # 1. Convert to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold
    _, mask = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 3. Noise Removal
    clean_kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    
    # 4. Expansion (C-Space Generation)
    dilation_diameter = int(robot_radius * 2)
    
    if dilation_diameter > 0:
        expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_diameter, dilation_diameter))
        mask_expanded = cv2.dilate(mask_cleaned, expansion_kernel)
    else:
        mask_expanded = mask_cleaned

    # 5. Find Contours (Vertices)
    #contours, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filter
    #valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, mask_expanded



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
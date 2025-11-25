import cv2
import numpy as np
import math
import sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry.base import JOIN_STYLE


#ArUco Initialization
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
except AttributeError:
    print("Error: ArUco dictionary not found. Is 'opencv-python-contrib' installed?")
    sys.exit()


def setup_camera(camera_index, width, height):
    """Initializes and configures the webcam."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION) # For mac iphone camera
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


def perspective_calibration(cap, map_width, map_height, matrix_file_path):
    """
    Calibrates the camera perspective by clicking on the four corners of the arena and returns the trasformation matrix.
    """
    src_points = []
    calibration_frame = None

    def _click_callback(event, x, y, flags, param):
        """Collect four clicks, then finish automatically."""
        nonlocal src_points, calibration_frame

        if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
            src_points.append((x, y))

            # Visual feedback
            cv2.circle(calibration_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click 4 Corners", calibration_frame)
            print(f"Point {len(src_points)}: ({x}, {y})")

            # If we have all 4 -> close window
            if len(src_points) == 4:
                print("All points selected. Computing transform...")
                cv2.destroyWindow("Click 4 Corners")

    #Capture frame to start calibration
    print("\n--- Perspective Calibration ---")
    print("Press 's' in the preview window to capture a calibration frame.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            return None
        preview = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Preview - Press 's' to capture", preview)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            calibration_frame = frame.copy()
            cv2.destroyWindow("Preview - Press 's' to capture")
            break

    #Click on four corners
    print("Click the arena corners in this order:")
    print("  1. Top-Left\n  2. Top-Right\n  3. Bottom-Right\n  4. Bottom-Left")
    cv2.namedWindow("Click 4 Corners")
    cv2.setMouseCallback("Click 4 Corners", _click_callback)
    cv2.imshow("Click 4 Corners", calibration_frame)

    #Wait until all 4 points are collected
    while len(src_points) < 4:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return None

    #Compute the transformation matrix
    src = np.float32(src_points)
    dst = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    np.save(matrix_file_path, matrix)
    print(f"Calibration complete. Saved to: {matrix_file_path}")
    
    return matrix


def detect_aruco_markers(frame):
    """
    Detects ALL ArUco markers in a frame.
    Returns: A dictionary {id: ((x, y), angle_deg)}
    """
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)

    poses = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners[0]
            #Get the center coordinates of each marker
            center_x, center_y = np.mean(pts, axis=0)

            #Compute the marker orientation in degrees
            top_mid = (pts[0] + pts[1]) / 2.0
            bottom_mid = (pts[2] + pts[3]) / 2.0
            dx = top_mid[0] - bottom_mid[0]
            dy = top_mid[1] - bottom_mid[1]
            angle_rad = math.atan2(dx, -dy)
            angle_deg = math.degrees(angle_rad) % 360  # wraps into [0, 360)

            poses[int(marker_id)] = ((int(center_x), int(center_y)), angle_deg)
    
    return poses


def mask_to_polygons(mask, min_area = 200, epsilon_factor = 0.01):
    """
    Convert a binary obstacle mask (white = obstacle) into polygons approximating each obstacle shape.
    """
    # Ensure binary mask
    _, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Find external contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Simplify contour into polygon
        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygons.append(approx.reshape(-1, 2).astype(np.int32))

    return polygons


def expand_and_merge_polygons(polygons, robot_radius, min_area = 10.0, simplify_tol = 1):
    """
    Expand polygons by robot_radius and merge polygons that overlap.
    """
    expanded = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        p = Polygon(poly)
        if not p.is_valid or p.area <= 0:
            continue
        # Expand the polygon
        p = p.buffer(robot_radius, join_style=JOIN_STYLE.mitre)
        # Simplify
        if simplify_tol > 0:
            p = p.simplify(simplify_tol, preserve_topology=True)
        if p.area >= min_area:
            expanded.append(p)
    if not expanded:
        return []
    # Merge overlapping shapes
    merged = unary_union(expanded)
    if merged.is_empty:
        return []
    # Handle Polygon vs MultiPolygon returned by the unary_union
    geoms = [merged] if merged.geom_type == "Polygon" else merged.geoms
    result = []
    for g in geoms:
        if g.area < min_area:
            continue
        coords = np.array(g.exterior.coords, dtype=np.float32)
        result.append(coords)

    return result

def remove_robot_from_mask(mask, thymio_pose, robot_clear_radius):
    """
    Zeros out a circular area around the robot so its LED is not treated as an obstacle.
    """
    if thymio_pose is None:
        return mask

    (x, y), _ = thymio_pose
    cleaned = mask.copy()
    cv2.circle(cleaned, (int(x), int(y)), int(robot_clear_radius), 0, thickness=-1)
    return cleaned

def detect_obstacles(frame, min_area, robot_radius, thymio_pose = None):
    """
    Detects obstacles (red/orange color) and returns the expanded polygon contours and the clean mask.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert the frame color scale from BGR to HSV
    #Define the ranges for obstacle color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([127, 59, 172])
    upper_red2 = np.array([179, 255, 255])
    #Create masks to detect pixels in these color ranges and combine them in one mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    #Remove noise by shrinking and expanding white regions
    clean_kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    #Remove thymio from the mask (red led problem)
    if thymio_pose is not None:
        mask_cleaned = remove_robot_from_mask(mask_cleaned, thymio_pose, robot_clear_radius=int(robot_radius * 1.2))

    #Approximate the obstacle shape with polygons and expand them by robot radius
    polygons = mask_to_polygons(mask = mask_cleaned, min_area= min_area)
    expanded_polygons = expand_and_merge_polygons(polygons, robot_radius)
    valid_contours = [poly.reshape(-1, 1, 2).astype(np.int32) for poly in expanded_polygons]
    
    return valid_contours, mask_cleaned


def draw_all_detections(frame, thymio_pose, goal_pos, obstacles):
    """
    Draws all detected elements onto the display frame.
    """
    if thymio_pose is not None:
        pos, angle = thymio_pose
        cv2.circle(frame, pos, 10, (0, 255, 255), -1) #yellow dot at the thymio's position
        angle_rad = math.radians(angle)
        end_x = int(pos[0] + 30 * math.sin(angle_rad))
        end_y = int(pos[1] - 30 * math.cos(angle_rad))
        cv2.arrowedLine(frame, pos, (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, "Thymio", 
                    (pos[0] + 15, pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if goal_pos is not None:
        cv2.circle(frame, goal_pos, 15, (255, 0, 0), -1) #blue dot at the goal position
        cv2.putText(frame, "Goal", (goal_pos[0] + 15, goal_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.drawContours(frame, obstacles, -1, (0, 0, 255), 2) #displays the obstacles dilated contours

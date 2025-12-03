import cv2
import numpy as np
import math
import sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE

# --- CONFIGURATION ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Corner Aruco indices for calibration
TOP_LEFT_INDEX = 2
TOP_RIGHT_INDEX = 3
BOTTOM_LEFT_INDEX = 4
BOTTOM_RIGHT_INDEX = 5

# --- HELPER FUNCTIONS ---

def _detect_aruco_markers(frame, detector):
    """
    Detects all Aruco markers in a frame.
    Returns {id: ((x, y), angle_deg)}.
    """
    corners, ids, _ = detector.detectMarkers(frame)
    poses = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners[0]
            center_x, center_y = np.mean(pts, axis=0)
            
            top_mid = (pts[0] + pts[1]) / 2.0
            bottom_mid = (pts[2] + pts[3]) / 2.0
            dx = top_mid[0] - bottom_mid[0]
            dy = top_mid[1] - bottom_mid[1]
            angle_rad = math.atan2(dx, -dy)
            angle_deg = math.degrees(angle_rad) % 360
            poses[int(marker_id)] = ((int(center_x), int(center_y)), angle_deg)
    return poses

def _perspective_calibration(cap, detector, map_width, map_height, matrix_file_path):
    """
    Automated calibration using 4 corner markers.
    """
    print(f"--- CALIBRATION MODE ---")
    print(f"Please place markers: {TOP_LEFT_INDEX} (TL), {TOP_RIGHT_INDEX} (TR), {BOTTOM_RIGHT_INDEX} (BR), {BOTTOM_LEFT_INDEX} (BL)")
    print("Press 's' to capture.")

    while True:
        ret, frame = cap.read()
        if not ret: return None

        preview = frame.copy()
        arucos = _detect_aruco_markers(preview, detector)

        corners_found = 0
        for marker_id, (pos, angle) in arucos.items():
            if marker_id in [TOP_LEFT_INDEX, TOP_RIGHT_INDEX, BOTTOM_LEFT_INDEX, BOTTOM_RIGHT_INDEX]:
                cv2.circle(preview, pos, 5, (0, 255, 0), -1)
                corners_found += 1

        status_text = "Ready to Calibrate" if corners_found == 4 else f"Found {corners_found}/4 Markers"
        color = (0, 255, 0) if corners_found == 4 else (0, 0, 255)
        cv2.putText(preview, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("ArUco Calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and corners_found == 4:
            calibration_frame = frame.copy()
            cv2.destroyWindow("ArUco Calibration")
            break

    # Calculate Matrix
    arucos = _detect_aruco_markers(calibration_frame, detector)
    
    # Verify markers exist in the captured frame
    if not all(k in arucos for k in [TOP_LEFT_INDEX, TOP_RIGHT_INDEX, BOTTOM_RIGHT_INDEX, BOTTOM_LEFT_INDEX]):
        print("Error: Markers lost during capture.")
        return None

    src = np.float32([
        arucos[TOP_LEFT_INDEX][0],
        arucos[TOP_RIGHT_INDEX][0],
        arucos[BOTTOM_RIGHT_INDEX][0],
        arucos[BOTTOM_LEFT_INDEX][0]
    ])

    dst = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    np.save(matrix_file_path, matrix)
    print(f"Calibration complete. Saved matrix to: {matrix_file_path}")
    return matrix

def _mask_to_polygons(mask, min_area=100, epsilon_factor=0.01):
    """Converts a binary mask into a list of polygon coordinates."""
    _, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx.reshape(-1, 2).astype(np.int32))
    return polygons

def _expand_and_merge_polygons(polygons_list, radius_px):
    """Uses Shapely to buffer (expand) polygons and merge overlaps."""
    shapely_polys = []
    for pts in polygons_list:
        try:
            p = Polygon(pts)
            if not p.is_valid: p = p.buffer(0)
            expanded = p.buffer(radius_px, join_style=JOIN_STYLE.mitre)
            shapely_polys.append(expanded)
        except Exception:
            continue

    if not shapely_polys: return []

    merged_geom = unary_union(shapely_polys)
    if merged_geom.is_empty: return []
    
    geoms = [merged_geom] if merged_geom.geom_type == 'Polygon' else merged_geom.geoms
    final_polys = []
    
    for geom in geoms:
        coords = np.array(geom.exterior.coords, dtype=np.int32)
        final_polys.append(coords.reshape(-1, 1, 2))
        
    return final_polys

def _detect_obstacles(frame, min_area, robot_radius, thymio_pose=None):
    """Detects obstacles (red/orange) and expands them."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red Thresholds
    lower_red = np.array([166, 87, 220])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Noise removal
    clean_kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    
    # Mask out Thymio
    if thymio_pose is not None:
        (x, y), _ = thymio_pose
        cv2.circle(mask_cleaned, (int(x), int(y)), int(robot_radius * 1.2), 0, thickness=-1)

    # Polygon processing
    polygons = _mask_to_polygons(mask_cleaned, min_area=min_area)
    expanded_polygons = _expand_and_merge_polygons(polygons, robot_radius)
    
    return expanded_polygons, mask_cleaned

def _draw_all_detections(frame, obstacles=None, thymio_pose=None, goal_pos=None):
    """
    Draws all detected elements onto the display frame.
    """
    if thymio_pose is not None:
        pos, angle = thymio_pose
        cv2.circle(frame, pos, 10, (0, 255, 255), -1) #Draw yellow dot at the thymio's position
        angle_rad = math.radians(angle)
        end_x = int(pos[0] + 30 * math.sin(angle_rad))
        end_y = int(pos[1] - 30 * math.cos(angle_rad))
        cv2.arrowedLine(frame, pos, (end_x, end_y), (0, 255, 255), 2) #Draw yellow arrow to visualize orientation
        cv2.putText(frame, "Thymio", 
                    (pos[0] + 15, pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) #Adds text "Thymio"
    if goal_pos is not None:
        cv2.circle(frame, goal_pos, 15, (255, 0, 0), -1) #blue dot at the goal position
        cv2.putText(frame, "Goal", (goal_pos[0] + 15, goal_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) #Adds text "Goal"
    if obstacles is not None and len(obstacles) > 0:
        cv2.drawContours(frame, obstacles, -1, (0, 0, 255), 2) #displays the obstacles dilated contours


# --- VISION CLASS ---

class Vision:
    def __init__(self, camera_index, width, height, map_width, map_height, matrix_path, thymio_id, goal_id, min_area=100):
        self.map_width = map_width
        self.map_height = map_height
        self.thymio_id = thymio_id
        self.goal_id = goal_id
        self.min_area = min_area
        self.matrix_path = matrix_path
        
        # Setup Camera
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            print("Camera failed to open.")

        # Setup ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        # Load Matrix
        self.matrix = self._load_or_calibrate()
        
        # Auto-calculate radius if not provided
        self.px_per_cm = float(self.map_width) / 100.0 
        self.robot_radius_px = 7 * self.px_per_cm 

    def _load_or_calibrate(self):
        try:
            return np.load(self.matrix_path)
        except FileNotFoundError:
            print("Calibration Matrix not found.")
            if self.cap is None: return None
            return _perspective_calibration(self.cap, self.detector, self.map_width, self.map_height, self.matrix_path)

    def get_warped_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return cv2.warpPerspective(frame, self.matrix, (self.map_width, self.map_height))

    def detect_aruco_raw(self, frame):
        return _detect_aruco_markers(frame, self.detector)

    def get_thymio_pose(self, frame):
        return self.detect_aruco_raw(frame).get(self.thymio_id)

    def get_goal_pos(self, frame):
        pos = self.detect_aruco_raw(frame).get(self.goal_id)
        return pos[0] if pos else None

    def detect_obstacles(self):
        """Public method to capture stable frame and run detection."""
        # Warmup
        frames = []
        for _ in range(50):
            f = self.get_warped_frame()
            if f is not None: frames.append(f)
        
        if not frames: return [], None
        stable_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
        
        pose = self.get_thymio_pose(stable_frame)
        return _detect_obstacles(stable_frame, self.min_area, self.robot_radius_px, pose)

    def draw(self, frame, obstacles=None, pose=None, goal=None, path=None, path_idx=0, state_text=""):
        # We call the exact function you requested
        _draw_all_detections(frame, obstacles, pose, goal)
        
        # We add the path drawing here since it wasn't in your snippet
        if path:
            for i in range(len(path)-1):
                cv2.line(frame, tuple(map(int, path[i])), tuple(map(int, path[i+1])), (0, 255, 0), 2)
            if path_idx < len(path):
                cv2.circle(frame, tuple(map(int, path[path_idx])), 6, (255, 0, 255), -1)
        
        # State Text
        if state_text:
            col = (0, 255, 0) if "GLOBAL" in state_text else (0, 0, 255)
            cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

    def release(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

def draw_covariance_ellipse(frame, P, mm_per_px, scale_factor=3.0):
    """
    Draws the XY uncertainty ellipse and Theta uncertainty wedge.
    """
    # Multiplier to make small errors visible to the human eye
    VISUAL_GAIN = 10.0 
    
    # Calculate Angle of the XY error
    angle = np.degrees(np.arctan2(P[1,1], P[0, 0]))
    
    # Calculate Axis Lengths (3-sigma * visual_gain)
    # We apply VISUAL_GAIN here to make it huge enough to see
    width = 2 * scale_factor * np.sqrt(P[0,0]) / mm_per_px * VISUAL_GAIN
    height = 2 * scale_factor * np.sqrt(P[1,1]) / mm_per_px * VISUAL_GAIN
    
    # Clamp minimum size so it doesn't vanish
    width = max(20, width) 
    height = max(20, height)
    
    # Fixed Position (Bottom Left)
    fixed_center = (200, frame.shape[0] - 200)
    
    try:
        # --- DRAW XY ELLIPSE (Blue) ---
        # We draw a filled semi-transparent ellipse if possible, but OpenCV
        cv2.ellipse(frame, fixed_center, (int(width/2), int(height/2)), 
                     int(angle), 0, 360, (255, 100, 0), 2)
        
        # --- DRAW THETA UNCERTAINTY (Yellow Wedge) ---
        var_theta = P[2, 2]
        if var_theta < 0: var_theta = 0
        std_theta = np.sqrt(var_theta)
        
        # Calculate angular spread (3-sigma)
        spread_deg = np.degrees(3 * std_theta)
        
        # Limit spread for visualization sanity
        spread_deg = min(spread_deg, 180)
        
        # Arrow Length (proportional to ellipse size or fixed)
        arrow_len = 100
        
        # Draw the "Cone" of uncertainty
        # We assume "Up" (-90 degrees in OpenCV) is the reference direction
        start_angle = -90 - spread_deg
        end_angle = -90 + spread_deg
        
        # Draw filled arc section
        cv2.ellipse(frame, fixed_center, (arrow_len, arrow_len), 0, start_angle, end_angle, (0, 255, 255), -1)
        
        # Draw the "Mean" Arrow (Center of the cone)
        p1 = fixed_center
        p2 = (fixed_center[0], fixed_center[1] - arrow_len)
        cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.2)

        # --- LABELS ---
        cv2.putText(frame, f"EKF Uncertainty (x{int(VISUAL_GAIN)})", (fixed_center[0]-100, fixed_center[1]+120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Theta: +/- {spread_deg:.1f} deg", (fixed_center[0]-80, fixed_center[1]+145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    except Exception as e:
        print(f"Viz Error: {e}")
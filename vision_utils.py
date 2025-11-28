import cv2
import numpy as np
import math
import sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry.base import JOIN_STYLE

#Corner Aruco indices for calibration
TOP_LEFT_INDEX = 2
TOP_RIGHT_INDEX = 3
BOTTOM_LEFT_INDEX = 4
BOTTOM_RIGHT_INDEX = 5

#ArUco Initialization
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
except AttributeError:
    print("Error: Aruco dictionary not found. Is 'opencv-python-contrib' installed?")
    sys.exit()


def _setup_camera(camera_index, width, height):
    """Initializes and configures the webcam."""
    #cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) #For provided webcam
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION) #For mac iphone camera
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {camera_index}.")
        return None
    #Set camera parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Webcam {camera_index} opened successfully.")
    return cap


def _load_transform_matrix(path):
    """Loads the perspective transform matrix from a .npy file."""
    try:
        matrix = np.load(path)
        print(f"Loaded perspective transform matrix from {path}.")
        return matrix
    except FileNotFoundError:
        print(f"Info: Matrix file not found at {path}.")
        return None


def _perspective_calibration(cap, map_width, map_height, matrix_file_path):
    """
    Creates the calibration matrix using the arucos as corner markers:
    first shows camera frame with detected markers, then waits for user to press 's' to validate
    and create the calibration matrix.
    Returns the matrix and saves it to indicated path.
    """
    calibration_frame = None

    #Wait for user to select the frame to use for calibration
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            return None

        preview = frame.copy()
        arucos = _detect_aruco_markers(preview)

        # Draw detected ArUco centers for feedback
        for marker_id, (pos, angle) in arucos.items():
            if marker_id in [TOP_LEFT_INDEX, TOP_RIGHT_INDEX, BOTTOM_LEFT_INDEX, BOTTOM_RIGHT_INDEX]:
                cv2.circle(preview, pos, 5, (0, 0, 255), -1)


        cv2.imshow("ArUco Calibration - Press 's' when ready", preview)
        key = cv2.waitKey(1) & 0xFF

        #If 's' key is pressed, the current frame is used to compute the matrix
        if key == ord('s'):
            calibration_frame = frame.copy()
            cv2.destroyWindow("ArUco Calibration - Press 's' when ready")
            break

    # Run detection on the captured frame
    arucos = _detect_aruco_markers(calibration_frame)

    # Extract the corner centers
    (top_left_center,  _ ) = arucos[TOP_LEFT_INDEX]
    (top_right_center, _ ) = arucos[TOP_RIGHT_INDEX]
    (bottom_right_center, _ ) = arucos[BOTTOM_RIGHT_INDEX]
    (bottom_left_center, _ ) = arucos[BOTTOM_LEFT_INDEX]

    src = np.float32([
        top_left_center,
        top_right_center,
        bottom_right_center,
        bottom_left_center
    ])

    dst = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])
    
    #Compute the matrix from the corners and distances
    matrix = cv2.getPerspectiveTransform(src, dst)
    np.save(matrix_file_path, matrix)
    print(f"Calibration complete. Saved matrix to: {matrix_file_path}")

    return matrix


def _detect_aruco_markers(frame):
    """
    Detects all Aruco markers in a frame.
    Returns a dictionary {id: ((x, y), angle_deg)}.
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


def _mask_to_polygons(mask, min_area = 200, epsilon_factor = 0.01):
    """
    Converts a binary obstacle mask (white = obstacle) into polygons approximating each obstacle shape.
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


def _expand_and_merge_polygons(polygons, robot_radius, min_area = 10.0, simplify_tol = 1):
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


def _detect_obstacles(frame, min_area, robot_radius, thymio_pose = None):
    """
    Detects obstacles (red/orange color) and returns the expanded polygon contours and the clean mask.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert the frame color scale from BGR to HSV
    #Define the ranges for obstacle color
    lower_red = np.array([0, 160, 64])
    upper_red = np.array([20, 255, 255])
    #Create masks to detect pixels in these color ranges and combine them in one mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #Remove noise by shrinking and expanding white regions
    clean_kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    #Remove thymio from the mask (red led problem) by adding black disk around thymio position
    if thymio_pose is not None:
        (x, y), _ = thymio_pose
        cv2.circle(mask_cleaned, (int(x), int(y)), int(robot_radius * 1.2), 0, thickness=-1)

    #Approximate the obstacle shape with polygons and expand them by robot radius to avoid collisions (defines the danger zone)
    polygons = _mask_to_polygons(mask = mask_cleaned, min_area= min_area)
    expanded_polygons = _expand_and_merge_polygons(polygons, robot_radius)
    valid_contours = [poly.reshape(-1, 1, 2).astype(np.int32) for poly in expanded_polygons]
    
    return valid_contours, mask_cleaned


def _draw_all_detections(frame, thymio_pose, goal_pos, obstacles):
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
    cv2.drawContours(frame, obstacles, -1, (0, 0, 255), 2) #displays the obstacles dilated contours



#Vision Class Definition

class Vision:

    def __init__(
        self,
        camera_index,
        cam_width,
        cam_height,
        map_width,
        map_height,
        matrix_file_path,
        thymio_marker_id=None,
        goal_marker_id=None,
        min_obstacle_area=200,
    ):
        self.camera_index = camera_index
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.map_width = map_width
        self.map_height = map_height
        self.matrix_file_path = matrix_file_path

        self.thymio_marker_id = thymio_marker_id
        self.goal_marker_id = goal_marker_id
        self.min_obstacle_area = min_obstacle_area

        #Opens the camera and sets parameters
        self.cap = _setup_camera(self.camera_index, self.cam_width, self.cam_height)

        # Use existing helper to load / compute matrix
        self.matrix = _load_transform_matrix(self.matrix_file_path)
        if self.matrix is None:
            print("No existing matrix found, running calibration...")
            self.matrix = _perspective_calibration(self.cap, self.map_width, self.map_height, self.matrix_file_path)
        
        #Compute the robot radius in pixels using known distance between corner arucos
        self._init_robot_radius_px()

    #Scale calculation method

    ###################################################################################
    ########## !!!! CHANGE DISTANCES TO 100 70 WHEN WE USE TOTAL MAP !!!! #############
    ###################################################################################

    def _init_robot_radius_px(self):
        """Compute the thymio's radius in pixels using known corner distances"""
        robot_radius_cm = 7 # Actual thymio radius from center of aruco
        aruco_corner_distance_cm_width = 70 #Distance between centers of arucos from top-left to bottom-left corners.
        aruco_corner_distance_cm_height = 50 #Distance between centers of arucos from top-left to top_right corners.
        px_per_cm_height = float(self.map_height)/aruco_corner_distance_cm_height #Compute height px/cm scale
        px_per_cm_width = float(self.map_width)/aruco_corner_distance_cm_width #Compute width px/cm scale
        px_per_cm = 0.5 * (px_per_cm_height + px_per_cm_width) #Compute average scale

        self.robot_radius_px = robot_radius_cm * px_per_cm
        
    #Frame access methods

    def get_raw_frame(self):
        """Returns a single raw frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        return frame

    def get_warped_frame(self):
        """
        Returns a perspective-warped frame using the calibration matrix.
        """
        frame = self.get_raw_frame()
        warped = cv2.warpPerspective(
            frame, self.matrix, (self.map_width, self.map_height)
        )
        return warped

    #Marker detection methods

    def detect_markers(self, frame=None):
        """
        Detects all ArUco markers in the given frame (or in a warped frame if not indicated).
        """
        if frame is None:
            frame = self.get_warped_frame()
        return _detect_aruco_markers(frame)

    def get_thymio_pose(self, frame=None):
        """
        Returns ((x, y), angle_deg) for thymio marker in the given frame (or in warped frame if not indicated).
        """
        if frame is None:
            frame = self.get_warped_frame()
        arucos = self.detect_markers(frame)
        return arucos.get(self.thymio_marker_id)

    def get_goal_pos(self, frame=None):
        """
        Returns (x, y) of goal marker center, or None if not found.
        """
        if self.goal_marker_id is None:
            return None
        if frame is None:
            frame = self.get_warped_frame()
        arucos = _detect_aruco_markers(frame)
        goal_pose = arucos.get(self.goal_marker_id)
        if goal_pose is None:
            return None
        (x, y), _ = goal_pose
        return (x, y)

    #Obstacle detection method

    def detect_obstacles(self):
        """
        Detects obstacles in a stable frame and returns obstacles and cleaned mask.
        """
        #Make sure the camera is running by skipping first five frames.
        for _ in range(5):
            self.get_warped_frame()
        frames = []
        for _ in range(5):
            frames.append(self.get_warped_frame())

        # Average them to reduce noise & flicker
        stable_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)

        #Try to detect the thymio in the stable frame
        thymio_pose = self.get_thymio_pose(stable_frame)
        
        #Detect the obstacles taking into account thymio pose
        obstacles, mask_cleaned = _detect_obstacles(
            stable_frame,
            min_area=self.min_obstacle_area,
            robot_radius=self.robot_radius_px,
            thymio_pose=thymio_pose,
        )
        return obstacles, mask_cleaned

    #Drawing method

    def draw(self, frame, thymio_pose, goal_pos, obstacles):
        """
        Draws all detections to visualize everything on a frame.
        """
        _draw_all_detections(frame, thymio_pose, goal_pos, obstacles)
        return frame

    #Release camera method

    def release(self):
        """Releases camera and destroy OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

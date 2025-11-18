import cv2
import numpy as np
import math

# --- ARUCO INITIALIZATION (done once) ---
# We initialize the detector here so it's ready to be used
# by the functions in this file.
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
except AttributeError:
    print("Error: ArUco dictionary not found. Is 'opencv-python-contrib' installed?")
    exit()

# --- UTILITY FUNCTIONS ---

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
        print(f"Error: Could not find matrix file at {path}.")
        print("Please run the Step 1 calibration script first.")
        return None

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
            
            # Calculate center
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Calculate orientation
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
    
    # Create a binary mask
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    # Clean up the mask to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, mask_cleaned

def draw_all_detections(frame, thymio_pose, goal_pos, obstacles, thymio_id, goal_id):
    """
    Draws all detected elements onto the display frame.
    Modifies the frame in-place.
    """
    # Draw Thymio
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

    # Draw Goal
    if goal_pos is not None:
        cv2.circle(frame, goal_pos, 15, (255, 0, 0), -1) # Blue
        cv2.putText(frame, f"Goal (ID {goal_id})", (goal_pos[0] + 15, goal_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw Obstacles
    cv2.drawContours(frame, obstacles, -1, (0, 0, 255), 2) # Red outlines
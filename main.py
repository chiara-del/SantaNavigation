import cv2
import numpy as np
import vision_utils as vu  # our utility file
import sys

# --- 1. CONFIGURATION ---

# Camera setting
CAMERA_INDEX = 1     #(0=personal webcam, 1=USB webcam)
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Map settings
MAP_WIDTH = 800
MAP_HEIGHT = 600
MATRIX_FILE_PATH = "my_matrix.npy"  # File to save/load calibration

# Marker IDs
THYMIO_MARKER_ID = 0  # ID of the ArUco marker on your Thymio
GOAL_MARKER_ID = 1    # ID of the ArUco marker for the goal

# Choose your method: "HSV" or "GRAY"
OBSTACLE_METHOD = "HSV" 

# HSV settings (used if OBSTACLE_METHOD == "HSV")
LOWER_WHITE_HSV = np.array([0, 0, 50])   
UPPER_WHITE_HSV = np.array([180, 25, 255])

# Grayscale settings (used if OBSTACLE_METHOD == "GRAY")
GRAYSCALE_THRESHOLD_VALUE = 150   

MIN_OBSTACLE_AREA = 200  # Minimum pixel area


def main():
    """
    Main application loop for robot localization.
    """
    
    # --- 2. INITIALIZATION ---
    
    cap = vu.setup_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    if cap is None:
        sys.exit("Initialization failed: Could not open camera.")

    matrix = vu.load_transform_matrix(MATRIX_FILE_PATH)
    
    if matrix is None:
        matrix = vu.run_calibration_wizard(
            cap, MAP_WIDTH, MAP_HEIGHT, MATRIX_FILE_PATH
        )
        if matrix is None:
            sys.exit("Calibration failed or was cancelled.")

    print("\nInitialization complete. Starting main localization loop...")
    print("Press 'q' in any window to quit.")

    # --- 3. MAIN LOOP (Step 2) ---

    try:
        while True:
            # 1. Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Exiting...")
                break
            
            # 2. DETECT ARUCO on the HIGH-RES frame *first*
            # We pass the *original* 'frame' here, not the warped map
            all_poses_original_coords = vu.detect_aruco_markers(frame)

            # 3. Apply Perspective Transform to the MAP
            #top_down_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))
            
            # 3. Detect all elements
            #all_poses = vu.detect_aruco_markers(frame) # Using the 'Best Fix'
            top_down_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))

            # 4. Detect Obstacles (using the chosen method)
            
            if OBSTACLE_METHOD == "HSV":
                obstacle_contours, obstacle_mask = vu.detect_obstacles_hsv(
                    top_down_map, LOWER_WHITE_HSV, UPPER_WHITE_HSV, MIN_OBSTACLE_AREA
                )
            elif OBSTACLE_METHOD == "GRAY":
                obstacle_contours, obstacle_mask = vu.detect_obstacles_grayscale(
                    top_down_map, GRAYSCALE_THRESHOLD_VALUE, MIN_OBSTACLE_AREA
                )
            else:
                print(f"Error: Unknown OBSTACLE_METHOD: {OBSTACLE_METHOD}")
                obstacle_contours = []
                obstacle_mask = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

            # 5. TRANSFORM ArUco coordinates into Map coordinates
            thymio_pose = None
            goal_position = None
            
            # Process Thymio
            if THYMIO_MARKER_ID in all_poses_original_coords:
                original_pose = all_poses_original_coords[THYMIO_MARKER_ID]
                original_pos, original_angle = original_pose
                
                # Transform the center point
                # Note: cv2.perspectiveTransform needs a 3D array: [[[x, y]]]
                original_point_np = np.float32([[original_pos]]).reshape(-1, 1, 2)
                transformed_point = cv2.perspectiveTransform(original_point_np, matrix)
                
                map_x = int(transformed_point[0][0][0])
                map_y = int(transformed_point[0][0][1])
                
                # We can't transform the angle directly, but it's often
                # good enough *unless* the camera is at a very sharp angle.
                # For a top-down camera, the angle is preserved.
                thymio_pose = ((map_x, map_y), original_angle)

            ##DEBUGGING
            #print('Thymio pose:', thymio_pose)

            # Process Goal
            if GOAL_MARKER_ID in all_poses_original_coords:
                original_pos, _ = all_poses_original_coords[GOAL_MARKER_ID]
                original_point_np = np.float32([[original_pos]]).reshape(-1, 1, 2)
                transformed_point = cv2.perspectiveTransform(original_point_np, matrix)
                goal_position = (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))

            ##DEBUGGING
            #print('Goal position:', goal_position)
            
            # 6. Visualization 
            display_frame = top_down_map.copy()
            vu.draw_all_detections(
                display_frame, thymio_pose, goal_position, 
                obstacle_contours, THYMIO_MARKER_ID, GOAL_MARKER_ID
            )
            
            cv2.imshow("Top-Down Map with Detections", display_frame)
            cv2.imshow("Obstacle Mask (Debug)", obstacle_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Cleaning up and closing windows.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
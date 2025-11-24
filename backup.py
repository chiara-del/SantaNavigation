# here I paste the previous main functon that does not transform the detection
def main():
    """
    Main application loop for robot localization.
    Runs Step 1 (Calibration) if needed, then runs Step 2 (Localization).
    """
    
    # --- 2. INITIALIZATION ---
    
    # Setup the camera (needed for both calibration and main loop)
    cap = vu.setup_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    if cap is None:
        sys.exit("Initialization failed: Could not open camera.")

    # Try to load the perspective transform matrix
    matrix = vu.load_transform_matrix(MATRIX_FILE_PATH)
    
    # If matrix doesn't exist, run the calibration wizard (Step 1)
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
            
            # 2. Apply Perspective Transform
            top_down_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))

            # 3. Detect all elements
            all_poses = vu.detect_aruco_markers(top_down_map)
            
            obstacle_contours, obstacle_mask = vu.detect_obstacles_hsv(
                top_down_map, LOWER_WHITE_HSV, UPPER_WHITE_HSV, MIN_OBSTACLE_AREA
            )

            # 4. Process Detections
            thymio_pose = all_poses.get(THYMIO_MARKER_ID)
            goal_pose = all_poses.get(GOAL_MARKER_ID)
            goal_position = goal_pose[0] if goal_pose else None

            # 5. Visualization
            display_frame = top_down_map.copy()
            vu.draw_all_detections(
                display_frame, thymio_pose, goal_position, 
                obstacle_contours, THYMIO_MARKER_ID, GOAL_MARKER_ID
            )
            
            cv2.imshow("Top-Down Map with Detections", display_frame)
            cv2.imshow("Obstacle Mask (Debug)", obstacle_mask)

            # 6. Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- 4. CLEANUP ---
        print("Cleaning up and closing windows.")
        cap.release()
        cv2.destroyAllWindows()





# BACKUP OF HSV AND GRAYSCALE (REPLACED BY RED OBSTACLE DETECTION)

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
    
    polygons = mask_to_polygons(mask_expanded)
    valid_contours = [poly.reshape(-1, 1, 2).astype(np.int32) for poly in polygons]

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

    # 4. Create polygons from obstacle shapes and get contours
    polygons = mask_to_polygons(mask_cleaned)

    # 5. Dilate the obstacle polygons by robot size
    expanded_polygons = expand_and_merge_polygons(polygons, robot_radius)

    valid_contours = [poly.reshape(-1, 1, 2).astype(np.int32) for poly in expanded_polygons]
    
    return valid_contours, mask_cleaned
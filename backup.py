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
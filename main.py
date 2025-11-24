import cv2
import numpy as np
import vision_utils as vu
import pathplanning_utils as pu
import control_utils as cu
import sys
import asyncio
import time  # <--- NEW IMPORT
from tdmclient import ClientAsync

# --- 1. CONFIGURATION ---
CAMERA_INDEX = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
MAP_WIDTH = 700
MAP_HEIGHT = 500
MATRIX_FILE_PATH = "calibration_matrix.npy"

THYMIO_MARKER_ID = 0
GOAL_MARKER_ID = 1

# Measurements
ROBOT_RADIUS_PX = 66 
MIN_OBSTACLE_AREA = 100

# Control Settings
WAYPOINT_REACH_THRESHOLD = 30
FORWARD_SPEED = 100
TURNING_GAIN_KP = 3.0

# Resilience Settings 
KIDNAPPING_THRESHOLD_PX = 60  # If robot is >60px from target, re-plan
MAX_BLIND_DURATION = 0.5      # Seconds to keep driving without vision

def transform_point(point, matrix):
    point_np = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point_np, matrix)
    return (int(transformed[0][0][0]), int(transformed[0][0][1]))

async def main():
    # --- 1. ROBOT CONNECTION (Explicit Method) ---
    client = ClientAsync()    
    print("Waiting for Thymio node...")
    node = await client.wait_for_node()
    await node.lock()
    print("Thymio Connected and Locked!")

    # Initialize our controller with this active node
    robot = cu.ThymioController(node)

    # --- 2. CAMERA SETUP ---
    cap = vu.setup_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    
    # Safety check if camera failed
    if cap is None: 
        await node.unlock()
        return

    matrix = vu.load_transform_matrix(MATRIX_FILE_PATH)
    if matrix is None:
        matrix = vu.perspective_calibration(
            cap, MAP_WIDTH, MAP_HEIGHT, MATRIX_FILE_PATH
        )
        if matrix is None:
            print("Error: Matrix not found. Run calibration separately first.")
            await node.unlock()
            sys.exit("Calibration failed or was cancelled.")
    
    print("\nInitialization complete. Starting main localization loop...")
    print("Kidnapping recovery enabled.")
    print("Press 'q' to quit.")
    
    calculated_path = None
    current_waypoint_index = 0

    # State Estimation Variables
    last_valid_pose = None
    last_valid_time = time.time()
    
    # Find obstacles once using a stable frame
    # Warm up camera
    for _ in range(5):
        cap.read()

    # Collect stable frames
    frames = []
    for _ in range(5):
        ret, f = cap.read()
        frames.append(f)
        await asyncio.sleep(0.05)

    # Average them to reduce noise & flicker
    stable_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
    top_down_map = cv2.warpPerspective(stable_frame, matrix, (MAP_WIDTH, MAP_HEIGHT))
    obstacle_contours, obstacle_mask = vu.detect_obstacles(top_down_map, MIN_OBSTACLE_AREA, ROBOT_RADIUS_PX)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Exiting...")
                break
            
            # --- VISION ---
            all_poses_raw = vu.detect_aruco_markers(frame)
            top_down_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))

            # Commented this to only detect obstacles at  the start
            # obstacle_contours, obstacle_mask = vu.detect_obstacles(
            #     top_down_map, 
            #     MIN_OBSTACLE_AREA, 
            #     ROBOT_RADIUS_PX
            # )
            
            # --- STATE ESTIMATION (Handle Lost Position) ---
            current_time = time.time()
            thymio_pose = None
            
            # Check for Thymio Marker
            if THYMIO_MARKER_ID in all_poses_raw:
                raw_pt, angle = all_poses_raw[THYMIO_MARKER_ID]
                map_pt = transform_point(raw_pt, matrix)
                thymio_pose = (map_pt, angle)
                
                # Update "Memory"
                last_valid_pose = thymio_pose
                last_valid_time = current_time
            
            elif last_valid_pose is not None:
                # If marker is missing, check how long we've been blind
                if current_time - last_valid_time < MAX_BLIND_DURATION:
                    # Trust the old position for a split second (Coast)
                    thymio_pose = last_valid_pose
                else:
                    # Too long! We have truly lost the robot.
                    thymio_pose = None

            # Check for Goal Marker
            goal_position = None
            if GOAL_MARKER_ID in all_poses_raw:
                raw_pt, _ = all_poses_raw[GOAL_MARKER_ID]
                goal_position = transform_point(raw_pt, matrix)

            # --- PLANNING & DYNAMIC RE-PLANNING (Handle Kidnapping) ---
            should_plan = False
            
            # Case 1: First Plan (We have robot & goal, but no path yet)
            if thymio_pose and goal_position and calculated_path is None:
                should_plan = True
                print("Initial Plan...")

            # Case 2: Kidnapping Check (We have a path, but robot is far off course)
            elif thymio_pose and goal_position and calculated_path:
                # Get current target waypoint
                if current_waypoint_index < len(calculated_path):
                    target = calculated_path[current_waypoint_index]
                    robot_xy = thymio_pose[0]
                    
                    # Calculate distance to where we are supposed to be going
                    dist_to_target = np.hypot(robot_xy[0]-target[0], robot_xy[1]-target[1])
                    
                    # If we are absurdly far from the target, we were kidnapped or pushed
                    if dist_to_target > KIDNAPPING_THRESHOLD_PX:
                        print(f"Kidnapping Detected! (Dist: {dist_to_target:.1f} px)")
                        should_plan = True

            # Execute Planning
            if should_plan:
                start_point = thymio_pose[0]
                planner_obstacles = [ [tuple(pt[0]) for pt in cnt] for cnt in obstacle_contours ]
                
                # Plan path
                path, _ = pu.plan_path(start_point, goal_position, planner_obstacles, safety=0.0)
                
                if path:
                    calculated_path = path
                    current_waypoint_index = 1 # Reset to first waypoint (skip start point)
                    print("Path Updated!")
                else:
                    print("Planning failed (blocked?). Stopping.")
                    calculated_path = None
                    await robot.stop()

            # --- CONTROL ---
            if thymio_pose and calculated_path and current_waypoint_index < len(calculated_path):
                robot_xy = thymio_pose[0]
                robot_angle = thymio_pose[1]
                target_waypoint = calculated_path[current_waypoint_index]
                
                if cu.check_waypoint_reached(robot_xy, target_waypoint, WAYPOINT_REACH_THRESHOLD):
                    print(f"Reached Waypoint {current_waypoint_index}")
                    current_waypoint_index += 1
                    
                    if current_waypoint_index >= len(calculated_path):
                        print("GOAL REACHED!")
                        await robot.stop()
                        calculated_path = None # Clear path to allow new goals
                else:
                    left, right = cu.calculate_control_command(
                        robot_xy, robot_angle, target_waypoint, 
                        base_speed=FORWARD_SPEED, k_p=TURNING_GAIN_KP
                    )
                    await robot.set_motors(left, right)

            elif thymio_pose is None:
                # Only stop if we have truly lost the robot for > 0.5 seconds
                await robot.stop()

            # --- VISUALIZATION ---
            if calculated_path:
                for i in range(len(calculated_path) - 1):
                    pt1 = (int(calculated_path[i][0]), int(calculated_path[i][1]))
                    pt2 = (int(calculated_path[i+1][0]), int(calculated_path[i+1][1]))
                    cv2.line(top_down_map, pt1, pt2, (0, 255, 0), 3)
                if current_waypoint_index < len(calculated_path):
                    target = calculated_path[current_waypoint_index]
                    cv2.circle(top_down_map, (int(target[0]), int(target[1])), 8, (0, 0, 255), -1)

            vu.draw_all_detections(
                top_down_map, thymio_pose, goal_position, obstacle_contours
            )
            
            cv2.imshow("Top-Down Map", top_down_map)
            cv2.imshow("Obstacle Mask", obstacle_mask)

            # Yield control to tdmclient
            await asyncio.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
    finally:
        print("Stopping robot...")
        await robot.stop()
        await node.unlock() # Unlock explicitly
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
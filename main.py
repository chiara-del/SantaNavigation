import cv2
import numpy as np
from vision_utils import Vision
import pathplanning_utils as pu
import control_utils as cu
import sys
import asyncio
import time
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
MIN_OBSTACLE_AREA = 100

# Control Settings
WAYPOINT_REACH_THRESHOLD = 30
FORWARD_SPEED = 100
TURNING_GAIN_KP = 3.0

# Local Avoidance Settings (Updated for Raw Values 0-4500)
OBST_THRL = 2000      # Lower threshold to exit avoidance (hysteresis)
OBST_THRH = 3000      # Higher threshold to enter avoidance
OBST_SPEEDGAIN = 0.06 # Gain for local avoidance
FOLLOW_PATH = 0
OBSTACLE_AVOIDANCE = 1

# Resilience Settings 
KIDNAPPING_CROSS_TRACK_THRESHOLD = 60  # Max allowed distance SIDEWAYS from the path line
MAX_BLIND_DURATION = 0.5               # Seconds to keep driving without vision

def get_distance_point_to_line(point, line_start, line_end):
    """
    Calculates the perpendicular distance (cross-track error) 
    from a point (robot) to the line segment (path).
    """
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    
    # Vector from line start to line end
    n = b - a
    # Length squared
    norm_n = np.linalg.norm(n)
    if norm_n == 0: return np.linalg.norm(p - a)
    
    # Distance formula: |(b-a) x (p-a)| / |b-a|
    dist = np.abs(np.cross(n, a - p)) / norm_n
    return dist

async def main():
    # Robot Connection
    client = ClientAsync()    
    print("Waiting for Thymio node...")
    node = await client.wait_for_node()
    await node.lock()
    print("Thymio Connected and Locked!")

    # Initialize our controller with this active node
    robot = cu.ThymioController(node)

    # Activate sensing from prox_sensors
    await node.wait_for_variables({"prox.horizontal"})
    state = FOLLOW_PATH 

    # Vision Setup
    vision = Vision(
        camera_index=CAMERA_INDEX,
        cam_width=CAMERA_WIDTH,
        cam_height=CAMERA_HEIGHT,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        matrix_file_path=MATRIX_FILE_PATH,
        thymio_marker_id=THYMIO_MARKER_ID,
        goal_marker_id=GOAL_MARKER_ID,
        min_obstacle_area=MIN_OBSTACLE_AREA,
    )
    
    # Initial Obstacle Detection
    obstacle_contours, obstacle_mask = vision.detect_obstacles()

    print("\nInitialization complete. Starting main localization loop...")
    print("Press 'q' to quit.")
    
    calculated_path = None
    current_waypoint_index = 0

    # State Estimation Variables
    last_valid_pose = None
    last_valid_time = time.time()

    try:
        while True:
            # Vision: get warped frame
            top_down_map = vision.get_warped_frame()

            # --- SENSOR PROCESSING ---
            # We must sum up sensors to see walls in front (index 2)
            vals = list(node["prox.horizontal"])
            
            # Left side intensity (Sensors 0, 1, and half of 2)
            prox_left = vals[0] + vals[1] + (vals[2] / 2)
            # Right side intensity (Sensors 4, 3, and half of 2)
            prox_right = vals[4] + vals[3] + (vals[2] / 2)
            
            # Max value seen by any front sensor (for state switching)
            max_prox = max(vals[:5]) 
            
            # --- STATE ESTIMATION (Handle Lost Position) ---
            current_time = time.time()
            thymio_pose = None
            
            thymio_pose_raw = vision.get_thymio_pose(top_down_map)
            
            if thymio_pose_raw is not None:
                thymio_pose = thymio_pose_raw
                last_valid_pose = thymio_pose
                last_valid_time = current_time
            elif last_valid_pose is not None:
                if current_time - last_valid_time < MAX_BLIND_DURATION:
                    thymio_pose = last_valid_pose
                else:
                    thymio_pose = None

            goal_position = vision.get_goal_pos(top_down_map)

            # --- PLANNING & DYNAMIC RE-PLANNING ---
            should_plan = False
            
            # Case 1: First Plan
            if thymio_pose and goal_position and calculated_path is None:
                should_plan = True
                print("Initial Plan...")

            # Case 2: Kidnapping Check (Using Cross-Track Error)
            elif thymio_pose and goal_position and calculated_path and state == FOLLOW_PATH:
                if current_waypoint_index < len(calculated_path):
                    target = calculated_path[current_waypoint_index]
                    
                    # Define the line segment we should be on
                    # If index is 0 or 1, previous point is start of path (or robot itself)
                    prev_idx = max(0, current_waypoint_index - 1)
                    prev_wp = calculated_path[prev_idx]
                    
                    robot_xy = thymio_pose[0]
                    
                    # Calculate how far SIDEWAYS we are from the ideal line
                    cross_track_error = get_distance_point_to_line(robot_xy, prev_wp, target)
                    
                    # Also check if we passed the target significantly (overshoot)
                    dist_to_target = np.hypot(robot_xy[0]-target[0], robot_xy[1]-target[1])
                    
                    # Trigger replan only if we are far off the line
                    if cross_track_error > KIDNAPPING_CROSS_TRACK_THRESHOLD:
                        print(f"Kidnapping Detected! (Off-track: {cross_track_error:.1f} px)")
                        should_plan = True

            # Execute Planning
            if should_plan:
                start_point = thymio_pose[0]
                planner_obstacles = [ [tuple(pt[0]) for pt in cnt] for cnt in obstacle_contours ]
                
                path, _ = pu.plan_path(start_point, goal_position, planner_obstacles, safety=0.0)
                
                if path:
                    calculated_path = path
                    current_waypoint_index = 1 
                    print("Path Updated!")
                else:
                    print("Planning failed (blocked?). Stopping.")
                    calculated_path = None
                    await robot.stop()

            # --- State Machine for Local Avoidance ---
            # Use max_prox (highest single sensor) to decide
            if state == FOLLOW_PATH: 
                if max_prox > OBST_THRH:  
                    state = OBSTACLE_AVOIDANCE
                    print("Switching to LOCAL AVOIDANCE")
            elif state == OBSTACLE_AVOIDANCE:
                if max_prox < OBST_THRL: 
                    state = FOLLOW_PATH
                    print("Switching to GLOBAL PATH")

            # --- CONTROL ---
            if state == FOLLOW_PATH:
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
                            calculated_path = None
                    else:
                        left, right = cu.calculate_control_command(
                            robot_xy, robot_angle, target_waypoint, 
                            base_speed=FORWARD_SPEED, k_p=TURNING_GAIN_KP
                        )
                        await robot.set_motors(left, right)
                
                elif thymio_pose is None:
                    await robot.stop()
                
            else:
                # --- LOCAL AVOIDANCE (Braitenberg) ---
                # Logic: If prox is high on LEFT, turn RIGHT (speed up Left motor)
                # Gain calculation: 0.06 * sensor_value (approx 0-250 speed addition)
                
                turn_left_force = prox_right * OBST_SPEEDGAIN
                turn_right_force = prox_left * OBST_SPEEDGAIN
                
                left_motor = int(FORWARD_SPEED + turn_right_force - turn_left_force)
                right_motor = int(FORWARD_SPEED + turn_left_force - turn_right_force)
                
                # Cap speeds
                left_motor = max(min(left_motor, 500), -500)
                right_motor = max(min(right_motor, 500), -500)
                
                await robot.set_motors(left_motor, right_motor)      
            
            # --- VISUALIZATION ---
            if calculated_path:
                for i in range(len(calculated_path) - 1):
                    pt1 = (int(calculated_path[i][0]), int(calculated_path[i][1]))
                    pt2 = (int(calculated_path[i+1][0]), int(calculated_path[i+1][1]))
                    cv2.line(top_down_map, pt1, pt2, (0, 255, 0), 3)
                if current_waypoint_index < len(calculated_path):
                    target = calculated_path[current_waypoint_index]
                    cv2.circle(top_down_map, (int(target[0]), int(target[1])), 8, (0, 0, 255), -1)

            vision.draw(top_down_map, obstacle_contours, thymio_pose, goal_position)
            
            # Visual feedback for state
            status_text = "GLOBAL" if state == FOLLOW_PATH else "LOCAL AVOIDANCE"
            color = (0, 255, 0) if state == FOLLOW_PATH else (0, 0, 255)
            cv2.putText(top_down_map, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Top-Down Map", top_down_map)
            # cv2.imshow("Obstacle Mask", obstacle_mask) # Optional

            await asyncio.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
    finally:
        print("Stopping robot...")
        await robot.stop()
        await node.unlock() 
        vision.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
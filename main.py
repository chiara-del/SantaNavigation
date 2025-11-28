import cv2
import numpy as np
from vision_utils import Vision
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
MIN_OBSTACLE_AREA = 100

# Control Settings
WAYPOINT_REACH_THRESHOLD = 30
FORWARD_SPEED = 100
TURNING_GAIN_KP = 3.0
OBST_THRL = 10      # low obstacle threshold to switch state 1->0
OBST_THRH = 20      # high obstacle threshold to switch state 0->1
OBST_SPEEDGAIN = 5  # /100 (actual gain: 5/100=0.05)
FOLLOW_PATH = 0
OBSTACLE_AVOIDANCE = 1

# Resilience Settings 
KIDNAPPING_THRESHOLD_PX = 60  # If robot is >60px from target, re-plan
MAX_BLIND_DURATION = 0.5      # Seconds to keep driving without vision


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
    state = FOLLOW_PATH          # 0=follow path, 1=obstacle avoidance

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
    
    # Detect the obstacles
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
            #Vision: get warped frame
            top_down_map = vision.get_warped_frame()

            # Get sensor values
            vals = list(node["prox.horizontal"])
            obst_left, obst_right = vals[0], vals[4]
            obst = [obst_left, obst_right]  # measurements from left and right prox sensors
            
            # --- STATE ESTIMATION (Handle Lost Position) ---
            current_time = time.time()
            thymio_pose = None
            
            # Check for Thymio Marker
            th_pose = vision.get_thymio_pose(top_down_map)
            if th_pose is not None:
                thymio_pose = th_pose
                
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
            g_pose = vision.get_goal_pos(top_down_map)
            if g_pose is not None:
                goal_position = g_pose

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

            # --- State Machine for Local Avoidance ---
            if state == FOLLOW_PATH: 
            # switch from goal tracking to obst avoidance if obstacle detected
                if (obst[0] > OBST_THRH):  # values higher if object near
                    state = OBSTACLE_AVOIDANCE
                elif (obst[1] > OBST_THRH):
                    state = OBSTACLE_AVOIDANCE
            elif state == OBSTACLE_AVOIDANCE:
                if obst[0] < OBST_THRL: #values lower if object far 
                    if obst[1] < OBST_THRL : 
                    # switch from obst avoidance to goal tracking if obstacle got unseen
                        state = FOLLOW_PATH

            # --- CONTROL ---
            if state == FOLLOW_PATH :
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
                
            else:
                left = FORWARD_SPEED + OBST_SPEEDGAIN * (obst[0] // 100)  #(left motor)
                right = FORWARD_SPEED + OBST_SPEEDGAIN * (obst[1] // 100) #(right motor)
                await robot.set_motors(left, right)      
            
            
                               
            # --- VISUALIZATION ---
            if calculated_path:
                for i in range(len(calculated_path) - 1):
                    pt1 = (int(calculated_path[i][0]), int(calculated_path[i][1]))
                    pt2 = (int(calculated_path[i+1][0]), int(calculated_path[i+1][1]))
                    cv2.line(top_down_map, pt1, pt2, (0, 255, 0), 3)
                if current_waypoint_index < len(calculated_path):
                    target = calculated_path[current_waypoint_index]
                    cv2.circle(top_down_map, (int(target[0]), int(target[1])), 8, (0, 0, 255), -1)

            vision.draw(top_down_map, thymio_pose, goal_position, obstacle_contours)
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
        vision.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
import cv2
import numpy as np
import vision_utils as vu
import pathplanning_utils as pu
import control_utils as cu
import sys
import asyncio
import time
from tdmclient import ClientAsync
from kalmann_filter import EKF2D

# --- CONFIG ---
CFG = { "CAM": 1, "RES": (1920, 1080), "MAP": (700, 500), "MTX": "calibration_matrix.npy",
        "IDS": (0, 1), "AREA": 100, "RAD": 41, "PX_CM": 7.3,
        "THRESH": (1000, 2000), "GAIN": 0.06, "BLIND": 0.5, "KIDNAP": 60 }

def check_kidnapping(pose, target, path, prev_idx, threshold):
    if not path: return False
    p, a, b = np.array(pose[0]), np.array(path[prev_idx]), np.array(target)
    n = b - a
    norm_n = np.linalg.norm(n)
    dist = np.abs(np.cross(n, a - p)) / norm_n if norm_n > 0 else np.linalg.norm(p - a)
    return dist > threshold

async def main():
    # 1. SETUP
    client = ClientAsync()
    try:
        print("Waiting for Thymio...")
        node = await client.wait_for_node()
        await node.lock()
        print("Connected!")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    robot = cu.ThymioController(node)
    follower = cu.PathFollower(speed=100, gain=3.0)
    await node.wait_for_variables({"prox.horizontal"})

    vision = vu.Vision(CFG["CAM"], *CFG["RES"], *CFG["MAP"], CFG["MTX"], 
                       *CFG["IDS"], CFG["AREA"], CFG["RAD"])
    if not vision.cap: 
        await node.unlock(); return
    
    #EKF init 
    Ts=0.01
    ekf = EKF2D(Ts=Ts,q_pos=0.2,q_vel=8.0,r_cam_posx=6.0, r_cam_posy=6.0, r_vel_cam_derived=120.0)
    seeded = False 
    last_cam = None # ((px,py),t) for velocity from camera


    # --- 2. ROBUST MAPPING (The Fix) ---
    print("Camera Warmup (2 seconds)...")
    
    # Force camera to run for 2 seconds to adjust exposure/color
    warmup_end = time.time() + 2.0
    while time.time() < warmup_end:
        vision.get_warped_frame()
        await asyncio.sleep(0.01)

    print("Mapping static obstacles...")
    obs_contours, mask = vision.detect_obstacles()
    
    # DEBUG: Show user what the robot detected
    if not obs_contours:
        print("⚠️ WARNING: No obstacles detected! Check lighting.")
    
    # Convert for planner
    planner_obs = [[tuple(pt[0]) for pt in cnt] for cnt in obs_contours]

    # Show the initial map for 1 second to verify
    debug_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(debug_view, "Initial Map Snapshot", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Map", debug_view)
    cv2.waitKey(1000) 

    # 3. LOOP
    print("Running... Press 'q' to quit.")
    state, last_pose, last_time = 0, None, time.time() # 0=Global, 1=Local

    try:
        while True:
            # SENSORS
            frame = vision.get_warped_frame()
            if frame is None: break
            prox = list(node["prox.horizontal"])
            
            #Predict EKF every loop 
            ekf.predict()

            # POSE
            pose = vision.get_thymio_pose(frame)
            if pose: last_pose, last_time = pose, time.time()
            elif time.time() - last_time < CFG["BLIND"]: pose = last_pose
            else: pose = None

            #EKF updates from camera
            if pose:
                (px,py), _ = pose

                #seed EKF on first valid pose 
                if not seeded:
                    ekf.x[:] = [px, py, 0.0, 0.0]
                    seeded = True
            
            #Position update 
            ekf.update_cam_pos(px,py)

            #Velocity from camera update
            if last_cam is not None:
                (px_prev,py_prev), t_prev = last_cam
                dt= time.time()-t_prev
                if dt > 1e-3:
                    vx = (px - px_prev) / dt
                    vy = (py - py_prev) / dt
                    ekf.update_vel_cam(vx, vy)
            
            last_cam = ((px,py), time.time())

            #choose pose for planner/control: EKF is seeded, else raw
            if seeded:
                x_hat, P_hat = ekf.get_state()
                ekf_pose = ((x_hat[0], x_hat[1]), None)
                pose_for_planner = ekf_pose
            else:
                pose_for_planner=pose

            goal = vision.get_goal_pos(frame)

            # PLANNING
            if pose_for_planner and goal:
                needs_plan = False
                if not follower.path: needs_plan = True
                elif follower.path and state == 0:
                    prev = max(0, follower.current_idx - 1)
                    target = follower.path[follower.current_idx]
                    if check_kidnapping(pose, target, follower.path, prev, CFG["KIDNAP"]):
                        print("Kidnapping detected!")
                        needs_plan = True

                if needs_plan:
                    print("Planning...")
                    path, _ = pu.plan_path(pose[0], goal, planner_obs, safety=0.0)
                    if path: follower.set_path(path); print("Path Found!")
                    else: print("No Path.")

            # STATE MACHINE
            max_prox = max(prox[:5])
            if state == 0 and max_prox > CFG["THRESH"][1]: state = 1
            elif state == 1 and max_prox < CFG["THRESH"][0]: state = 0

            # CONTROL
            if pose:
                if state == 0: # Global
                    l, r, done = follower.get_command(pose_for_planner)
                    if done: print("Goal!"); follower.path = None
                    await robot.set_motors(l, r)
                else: # Local
                    l, r = cu.calculate_avoidance_commands(prox, 100, CFG["GAIN"])
                    await robot.set_motors(l, r)
            else:
                await robot.stop()

            # VISUALIZATION
            vision.draw(frame, obs_contours, pose, goal, follower.path, follower.current_idx)
            # Visual feedback for state
            status_text = "GLOBAL" if state == 0 else "LOCAL AVOIDANCE"
            color = (0, 255, 0) if state == 0 else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Map", frame)

            #Draw EKF state explicitly (blue dot)
            if seeded:
                ex, ey = int(ekf.x[0]), int(ekf.x[1])
                cv2.circle(frame, (ex, ey), 5, (255, 0, 0), -1)
                cv2.putText(frame, "EKF", (ex+6, ey-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow("Map", frame)
            
            await asyncio.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        await robot.stop()
        await node.unlock()
        vision.release()

if __name__ == "__main__":
    asyncio.run(main())
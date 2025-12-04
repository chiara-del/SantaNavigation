import cv2
import numpy as np
import vision_utils as vu
import pathplanning_utils as pu
import control_utils as cu
import asyncio
import time
import math
from tdmclient import ClientAsync
from ekf_pose import EKFPose   # <- classe ci-dessus

# --- CONFIG ---
CFG = {
    "CAM": 0, "RES": (1920, 1080), "MAP": (1000, 700), "MTX": "calibration_matrix.npy",
    "IDS": (0, 1), "AREA": 100,
    "THRESH": (600, 1000), "GAIN": 0.06, "BLIND": 0.5, "KIDNAP": 60
}

async def main():
    #1) Connect to Thymio
    client = ClientAsync()
    try:
        print("Waiting for Thymio...")
        node = await client.wait_for_node()
        await node.lock()
        print("Connected!")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # Ensure we can read motor speeds
    await node.wait_for_variables({"prox.horizontal", "motor.left.speed", "motor.right.speed"})

    robot = cu.ThymioController(node)
    follower = cu.PathFollower(speed=100, gain=2.5)

    #2) Initialize Vision
    vision = vu.Vision(CFG["CAM"], *CFG["RES"], *CFG["MAP"], CFG["MTX"],
                       *CFG["IDS"], CFG["AREA"])
    if not vision.cap:
        await node.unlock()
        return

    #3) EKF initialization (units: mm, mm/s, rad)
    mm_per_px = 10.0/(vision.px_per_cm)   # mm/px 
    ekf = EKFPose()
    seeded = False

    # Speed conversions
    SPEED_TO_MMS = 0.33 #mm/s per Thymio unit
    b_mm = 95.0                          # track width (mm) à mesurer
    omega_scale = SPEED_TO_MMS / b_mm    # rad/s per (Thymio unit)

    #4) Obstacle Detection
    print("Camera Warmup (2 seconds)...")
    warmup_end = time.time() + 2.0
    while time.time() < warmup_end:
        vision.get_warped_frame()
        await asyncio.sleep(0.01)

    print("Mapping static obstacles...")
    obs_contours, mask = vision.detect_obstacles()
    if not obs_contours:
        print("WARNING: No obstacles detected! Check lighting.")
    planner_obs = [[tuple(pt[0]) for pt in cnt] for cnt in obs_contours]
    debug_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(debug_view, "Initial Map Snapshot", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Map", debug_view)
    cv2.waitKey(1500)

    # 5) Main Loop
    print("Running... Press 'q' to quit.")
    state = 0  # 0=Global, 1=Local

    try:
        while True:
            frame = vision.get_warped_frame()
            if frame is None:
                break
            prox = list(node["prox.horizontal"])

            # Build u = [v_meas, omega_meas] from wheels
            sL = float(node["motor.left.speed"])
            sR = float(node["motor.right.speed"])
            v_meas = 0.5 * (sL + sR) * SPEED_TO_MMS           # mm/s
            omega_meas = -(sR - sL) * omega_scale              # rad/s
            u = np.array([v_meas, omega_meas], float)

            # EKF predict with u
            ekf.predict(u)

            # Get Thymio pose from vision
            pose = vision.get_thymio_pose(frame)

            # EKF updates (position + theta)
            if pose:
                (px, py), angle_deg = pose
                x_mm = px * mm_per_px
                y_mm = py * mm_per_px
                theta_rad = math.radians(angle_deg)
            
                if not seeded:
                    ekf.x[:] = [x_mm, y_mm, theta_rad]
                    seeded = True

                ekf.update_pos(x_mm, y_mm)
                ekf.update_theta(theta_rad)

            # Planner/control pose (convert back to pixels)
            if seeded:
                x_hat, _ = ekf.get_state()
                px_hat = int(x_hat[0] / mm_per_px)
                py_hat = int(x_hat[1] / mm_per_px)
                theta_hat = (math.degrees(x_hat[2]))%360
                pose_for_planner = ((px_hat, py_hat), theta_hat)
            else:
                pose_for_planner = pose

            #Get Goal Position from vision
            goal = vision.get_goal_pos(frame)

            # State machine (local avoidance)
            max_prox = max(prox[:5])
            if state == 0 and max_prox > CFG["THRESH"][1]:
                state = 1
            elif state == 1 and max_prox < CFG["THRESH"][0]:
                state = 0
                follower.path = None

            # Planning
            if pose_for_planner and goal:
                needs_plan = False
                if not follower.path:
                    needs_plan = True
                elif follower.path and state == 0:
                    prev = max(0, follower.current_idx - 1)
                    target = follower.path[follower.current_idx]
                    if pu.check_kidnapping(pose_for_planner, target, follower.path, prev, CFG["KIDNAP"]):
                        print("Kidnapping detected -> Replan")
                        needs_plan = True

                if needs_plan:
                    print("Planning...")
                    path, _ = pu.plan_path(pose_for_planner[0], goal, planner_obs, safety=0.0)
                    
                    if path:
                        follower.set_path(path)
                        print("Path Found!")
                    else:
                        print("No Path.")



            # Control
            if pose_for_planner:
                if state == 0:
                    l, r, done = follower.get_command(pose_for_planner)
                    if done:
                        print("Goal reached!")
                        follower.path = None
                    await robot.set_motors(l, r)
                else:
                    l, r = cu.calculate_avoidance_commands(prox, 50, 1.8)
                    await robot.set_motors(l, r)
            else:
                await robot.stop()

            # Visualization
            # Main map Visualization
            status_text = "GLOBAL" if (state == 0) else "LOCAL AVOIDANCE"
            vision.draw(frame = frame,
                        obstacles = obs_contours, 
                        pose = pose, 
                        kalman_pose = pose_for_planner, 
                        goal = goal, 
                        path = follower.path, 
                        path_idx = follower.current_idx, 
                        state_text = status_text,
                        )
            cv2.imshow("Map", frame)

            # Kalman Visualization
            kalman_var = np.ones((400, 400, 3), dtype=np.uint8) * 255
            cv2.namedWindow("KalmanVariance", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("KalmanVariance", 400, 400)
            if seeded:
                ekf.draw_covariance_ellipse(kalman_var, mm_per_px)
            cv2.imshow("KalmanVariance", kalman_var)


            await asyncio.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        await robot.stop()
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
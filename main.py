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
    "CAM": 1, "RES": (1920, 1080), "MAP": (1000, 700), "MTX": "calibration_matrix.npy",
    "IDS": (0, 1), "AREA": 100, "RAD": 41, "PX_CM": 7.3,
    "THRESH": (1000, 2000), "GAIN": 0.06, "BLIND": 0.5, "KIDNAP": 60
}

def check_kidnapping(pose, target, path, prev_idx, threshold):
    if not path:
        return False
    p, a, b = np.array(pose[0]), np.array(path[prev_idx]), np.array(target)
    n = b - a
    norm_n = np.linalg.norm(n)
    dist = np.abs(np.cross(n, a - p)) / norm_n if norm_n > 0 else np.linalg.norm(p - a)
    return dist > threshold

async def main():
    # 1) CONNECT
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

    vision = vu.Vision(CFG["CAM"], *CFG["RES"], *CFG["MAP"], CFG["MTX"],
                       *CFG["IDS"], CFG["AREA"], CFG["RAD"])
    if not vision.cap:
        await node.unlock()
        return

    # 2) EKF INIT (units: mm, mm/s, rad)
    Ts = 0.01
    mm_per_px = 10.0 / CFG["PX_CM"]     # mm/px
    ekf = EKFPose(
        Ts=Ts,
        # Q (process): à affiner avec tes estimations converties
        q_x=32.813, q_y=0.001, q_theta=4e-3, q_v=1817.0,
        # R (measure): idem
        r_pos_x=0.208, r_pos_y=0.0,                # mm^2
        r_theta=3.817e-06,                # rad^2
        r_v=57.402                                  # (mm/s)^2
    )
    seeded = False

    # Speed conversions
    SPEED_TO_MMS = 0.43478260869565216   # mm/s per Thymio unit (Ex. 8)
    b_mm = 95.0                          # track width (mm) à mesurer
    omega_scale = SPEED_TO_MMS / b_mm    # rad/s per (Thymio unit)

    # 3) CAMERA WARMUP + STATIC MAP
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
    cv2.waitKey(1000)

    # 4) LOOP
    print("Running... Press 'q' to quit.")
    state, last_pose, last_time = 0, None, time.time()  # 0=Global, 1=Local

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
            omega_meas = (sR - sL) * omega_scale              # rad/s
            u = np.array([v_meas, omega_meas], float)

            # EKF predict with u
            ekf.predict(u)

            # Pose from vision
            pose = vision.get_thymio_pose(frame)
            now = time.time()
            if pose:
                last_pose, last_time = pose, now
            elif now - last_time < CFG["BLIND"]:
                pose = last_pose
            else:
                pose = None

            # EKF updates (position + theta)
            print('pose:', pose)
            if pose:
                (px, py), angle_deg = pose
                x_mm = px * mm_per_px
                y_mm = py * mm_per_px
                theta_rad = math.radians(angle_deg)
            
                if not seeded:
                    ekf.x[:] = [x_mm, y_mm, theta_rad, v_meas]
                    seeded = True

                ekf.update_pos(x_mm, y_mm)
                ekf.update_theta(theta_rad)
                # Optionnel: recaler v aussi (sinon déjà injecté via u)
                # ekf.update_v(v_meas)

            # Planner/control pose (convert back to pixels)
            if seeded:
                x_hat, P_hat = ekf.get_state()
                px_hat = int(x_hat[0] / mm_per_px)
                py_hat = int(x_hat[1] / mm_per_px)
                theta_hat = (math.degrees(x_hat[2]))%360
                pose_for_planner = ((px_hat, py_hat), theta_hat)
            else:
                pose_for_planner = pose
            print('pose_for_planner:', pose_for_planner)
            goal = vision.get_goal_pos(frame)

            # Planning
            if pose_for_planner and goal:
                needs_plan = False
                if not follower.path:
                    needs_plan = True
                elif follower.path and state == 0:
                    prev = max(0, follower.current_idx - 1)
                    target = follower.path[follower.current_idx]
                    if check_kidnapping(pose_for_planner, target, follower.path, prev, CFG["KIDNAP"]):
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

            # State machine (local avoidance)
            max_prox = max(prox[:5])
            if state == 0 and max_prox > CFG["THRESH"][1]:
                state = 1
            elif state == 1 and max_prox < CFG["THRESH"][0]:
                state = 0

            # Control
            if pose_for_planner:
                if state == 0:
                    l, r, done = follower.get_command(pose_for_planner)
                    if done:
                        print("Goal reached!")
                        follower.path = None
                    await robot.set_motors(l, r)
                else:
                    l, r = cu.calculate_avoidance_commands(prox, 50, CFG["GAIN"])
                    await robot.set_motors(l, r)
            else:
                await robot.stop()

            # Visualization (draw EKF pose in blue)
            vision.draw(frame, obs_contours, pose_for_planner, goal, follower.path, follower.current_idx)
            status_text = "GLOBAL (EKF u=[v,omega])" if (state == 0) else "LOCAL AVOIDANCE"
            color = (0, 255, 0) if state == 0 else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if seeded:
                ex, ey = int(px_hat), int(py_hat)
                cv2.circle(frame, (ex, ey), 5, (255, 0, 0), -1)
                cv2.putText(frame, "EKF", (ex+6, ey-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow("Map", frame)

            await asyncio.sleep(Ts)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        await robot.stop()
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
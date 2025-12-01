import cv2, time, numpy as np, asyncio
from vision_utils import Vision
from tdmclient import ClientAsync
import control_utils as cu  # must provide ThymioController(node).set_motors(l, r)

# Camera/map config
CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT = 1, 1920, 1080
MAP_WIDTH, MAP_HEIGHT = 700, 500
MATRIX_FILE_PATH = "calibration_matrix.npy"
THYMIO_MARKER_ID = 0

# Logging
DURATION = 18.0   # seconds total
Ts = 0.01         # loop sleep
ALPHA = 0.5       # r_v = ALPHA * Var(v_meas)


#measurement noise variance of speed measurement = how noisy speed sensor is --> large rv means trust speed measurement less and small rv makes it trust more 
#drive robot at roughly steady speed and compute speed v from consecutive camera positions 
#take a steady segment (middle of the run), compute total variance var(v_meas) over segement with alpha=0.5
#since we only observe total speed variability we split the variance between qvel and rv 

# Straight speed steps (keeps heading roughly constant)
async def motion_profile_rv(robot):
    await robot.set_motors(0, 0); await asyncio.sleep(1.0)
    for l, r, t in [(220,220,3.0), (280,280,3.0), (240,240,3.0)]:
        await robot.set_motors(l, r); await asyncio.sleep(t)
        await robot.set_motors(0, 0); await asyncio.sleep(0.6)
    await robot.set_motors(0, 0); await asyncio.sleep(1.0)

async def run():
    print("Estimating r_v from total speed variance. Robot will do straight speed steps.")
    # Connect Thymio
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    robot = cu.ThymioController(node)

    # Camera
    vision = Vision(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
                    MAP_WIDTH, MAP_HEIGHT, MATRIX_FILE_PATH,
                    thymio_marker_id=THYMIO_MARKER_ID,
                    goal_marker_id=9999, min_obstacle_area=100)

    xs, ys, ts = [], [], []
    t0 = time.time()

    # start motion concurrently
    motion_task = asyncio.create_task(motion_profile_rv(robot))

    try:
        while time.time() - t0 < DURATION:
            top = vision.get_warped_frame()
            pose = vision.get_thymio_pose(top)
            if pose is not None:
                (px, py), _ = pose
                xs.append(px); ys.append(py); ts.append(time.time())
                cv2.circle(top, (int(px), int(py)), 6, (0,255,0), -1)
            cv2.putText(top, f"samples: {len(xs)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("r_v collection", top)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(Ts)

        await motion_task
        await robot.set_motors(0, 0)
    finally:
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

    if len(xs) < 20:
        print("Not enough data. Keep the robot in view and try again.")
        return

    xs = np.array(xs, float); ys = np.array(ys, float); ts = np.array(ts, float)

    # Speed magnitude v (pixels/s)
    dt = np.diff(ts)
    dx = np.diff(xs); dy = np.diff(ys)
    mask = dt > 1e-3
    v = np.sqrt((dx[mask]/dt[mask])**2 + (dy[mask]/dt[mask])**2)
    if v.size < 15:
        print("Not enough valid velocity samples.")
        return

    # Middle 60% to avoid start/stop transients
    n = v.size
    v_seg = v[int(0.2*n):int(0.8*n)] if n >= 10 else v

    total_var = float(np.var(v_seg, ddof=1))
    r_v = float(max(1e-8, ALPHA * total_var))
    q_v= float(max(1e-8, ALPHA * total_var))

    print("\nEstimated r_v (speed measurement noise):")
    print(f"  Var(v_meas) = {total_var:.6f} (pixels/s)^2")
    print(f"  r_v = {r_v:.6f} with alpha = {ALPHA}")
    print(f"  q_v = {q_v:.6f} with alpha = {ALPHA}")
    
    # reconstruct mid-times for v
    t_v = 0.5*(ts[:-1] + ts[1:])[mask]
    i0, i1 = int(0.2*len(v)), int(0.8*len(v))

    plt.figure("Speed over time")
    plt.plot(t_v, v, label="|v| (pixels/s)")
    if i1 > i0:
        plt.axvspan(t_v[i0], t_v[i1-1], color='orange', alpha=0.2, label="steady segment")
    plt.xlabel("time (s)"); plt.ylabel("pixels/s"); plt.grid(True); plt.legend()

    plt.figure("Velocity increments Δv")
    plt.plot(dvx, label="Δv_x"); plt.plot(dvy, label="Δv_y")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("step"); plt.ylabel("pixels/s per step"); plt.grid(True); plt.legend()
    plt.title(f"Var(Δv_x)={q_vx:.4f}, Var(Δv_y)={q_vy:.4f} -> q_v≈{q_v_est:.4f}")

if __name__ == "__main__":
    asyncio.run(run())
import cv2, time, numpy as np, matplotlib.pyplot as plt
from vision_utils import Vision
import asyncio

# ADD: Thymio control
from tdmclient import ClientAsync
import control_utils as cu

CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT = 1, 1920, 1080
MAP_WIDTH, MAP_HEIGHT = 700, 500
MATRIX_FILE_PATH = "calibration_matrix.npy"
THYMIO_MARKER_ID = 0
Ts = 0.01
DURATION = 25.0

#we compute speed between frames 
#we predict the next postion by p_next=p_prev+v*dt
#compare prediction vs camera : error=measured postion - predicted position
#turning, slip, timing... make errors grow 
#we look at how the error variance increases step after step and fit a straight line 
#the slope of that line=q_pos=how much position uncertainty increases per step 


# ADD: motion helper for q_pos (figure-8)
async def motion_profile_qpos(robot):
    await robot.set_motors(0, 0); await asyncio.sleep(1.0)
    fwd, turn, seg_t = 300, 70, 3.0
    # left arc, right arc, repeat
    await robot.set_motors(fwd - turn, fwd + turn); await asyncio.sleep(seg_t)
    await robot.set_motors(fwd + turn, fwd - turn); await asyncio.sleep(seg_t)
    await robot.set_motors(fwd - turn, fwd + turn); await asyncio.sleep(seg_t)
    await robot.set_motors(fwd + turn, fwd - turn); await asyncio.sleep(seg_t)
    # short straight
    await robot.set_motors(fwd, fwd); await asyncio.sleep(2.0)
    await robot.set_motors(0, 0); await asyncio.sleep(1.5)

async def run_test():
    # Thymio connect
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

    # ADD: launch motion concurrently
    motion_task = asyncio.create_task(motion_profile_qpos(robot))

    try:
        while time.time() - t0 < DURATION:
            top = vision.get_warped_frame()
            pose = vision.get_thymio_pose(top)
            if pose is not None:
                (px, py), _ = pose
                xs.append(px); ys.append(py); ts.append(time.time())
                cv2.circle(top, (int(px), int(py)), 6, (0,255,0), -1)
            cv2.imshow("q_pos collection (raw)", top)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(Ts)

        await motion_task
        await robot.set_motors(0, 0)

    finally:
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

    xs = np.array(xs, float); ys = np.array(ys, float); ts = np.array(ts, float)
    if len(xs) < 20:
        print("Not enough data."); return

    # Velocities from positions
    vx = np.diff(xs) / np.diff(ts)  #vx[k]=(xs[k+1]-xs[k])/(ts[k+1]-ts[k])
    vy = np.diff(ys) / np.diff(ts)  #vy[k]=(ys[k+1]-ys[k])/(ts[k+1]-ts[k])

    # Predict positions with constant-velocity integration
    px_pred = [xs[0]]; py_pred = [ys[0]]
    for k in range(len(vx)):
        dt = ts[k+1] - ts[k]
        px_pred.append(px_pred[-1] + vx[k]*dt)  #integrate velocity each step
        py_pred.append(py_pred[-1] + vy[k]*dt)
    px_pred = np.array(px_pred); py_pred = np.array(py_pred)

    # Residuals
    ex = xs - px_pred
    ey = ys - py_pred

    # Approximate per-step variance growth (slope)
    idx = np.arange(len(ex))
    var_proxy = 0.5*((ex - np.mean(ex))**2 + (ey - np.mean(ey))**2) #squared error 
    var_smooth = np.convolve(var_proxy, np.ones(5)/5.0, mode='same')
    A = np.vstack([idx, np.ones_like(idx)]).T
    slope, intercept = np.linalg.lstsq(A, var_smooth, rcond=None)[0]
    q_pos = float(max(slope, 1e-4))
    #np.save(SAVE_PATH, np.array([q_pos]))

    print("\nq_pos (px^2 per step), raw:")
    print(f"  q_pos = {q_pos:.4f}")
    #print(f"Saved to {SAVE_PATH}")

    # Plots
    plt.figure("Position residuals (raw)")
    plt.plot(ex, label="e_x"); plt.plot(ey, label="e_y"); plt.grid(True); plt.legend()
    plt.figure("Residual variance growth (raw)")
    plt.plot(var_smooth, label="variance proxy"); plt.grid(True); plt.legend()
    plt.title(f"q_pos ≈ {q_pos:.4f}")
    plt.show()

if __name__ == "__main__":
    asyncio.run(run_test())
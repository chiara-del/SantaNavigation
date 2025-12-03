import cv2, time, numpy as np
from vision_utils import Vision

CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT = 1, 1920, 1080
MAP_WIDTH, MAP_HEIGHT = 700, 500
MATRIX_FILE_PATH = "calibration_matrix.npy"
THYMIO_MARKER_ID = 0
N_SAMPLES = 500


#R_pos=how noisy the camera's postion measurement is
#fix scene: keep robot perfectly still and use pipeline to warp frames to the map frame
#collect N frames and read the detected position (px,py) in map pixels
#store the positions in two arrays for posx and posy then take r_px and r_py as the variance of px and py respectively

async def run():
    print("Test A (raw): estimating R_pos (no outlier removal). Keep marker still.")
    
    # Connect Thymio
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()

    vision = Vision(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
                    MAP_WIDTH, MAP_HEIGHT, MATRIX_FILE_PATH,
                    thymio_marker_id=THYMIO_MARKER_ID,
                    goal_marker_id=9999, min_obstacle_area=100)

    xs, ys = [], []
    try:
        while len(xs) < N_SAMPLES:
            top = vision.get_warped_frame()
            pose = vision.get_thymio_pose(top)
            if pose is not None:
                (px, py), _ = pose
                xs.append(px); ys.append(py)
                cv2.circle(top, (int(px), int(py)), 6, (0,255,0), -1)
            cv2.putText(top, f"samples {len(xs)}/{N_SAMPLES}", (10,30), 0, 1, (0,255,0), 2)
            cv2.imshow("R_pos collection (raw)", top)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(0.01)
    finally:
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

    xs = np.array(xs, float); ys = np.array(ys, float)
    r_px = float(np.var(xs, ddof=1))
    r_py = float(np.var(ys, ddof=1))
    R_pos = np.diag([r_px, r_py])
    np.save(SAVE_PATH, R_pos)

    print("\nR_pos (pixels^2, raw):")
    print(f"  r_px = {r_px:.3f}, r_py = {r_py:.3f}")
    print(R_pos)
    

    # Plots
    plt.figure("Positions px and py")
    plt.plot(xs, label="px"); plt.plot(ys, label="py")
    plt.title("Static position samples (raw)"); plt.xlabel("sample #"); plt.ylabel("pixels"); plt.legend(); plt.grid(True)


if __name__ == "__main__":
    asyncio.run(run())
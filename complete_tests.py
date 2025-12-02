import cv2, time, math, numpy as np, asyncio
from tdmclient import ClientAsync
import vision_utils as vu  # ta classe Vision

# ---------- Camera/map config ----------
CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT = 1, 1920, 1080
MAP_WIDTH, MAP_HEIGHT = 1000, 700
MATRIX_FILE_PATH = "calibration_matrix.npy"
THYMIO_MARKER_ID = 0

# ---------- Timings ----------
Ts = 0.01
DUR_STATIC  = 25.0      # s (statique)
DUR_MOTION  = 25.0     # s (profil mouvement pour q_v, r_v, q_theta)
DUR_QPOS    = 25.0     # s (prédiction pure pour q_x, q_y)

# ---------- Units & conversions ----------
PX_CM = 7.3
mm_per_px = 10.0 / PX_CM              # mm/px
SPEED_TO_MMS = 0.43478260869565216    # mm/s per Thymio unit (Ex. 8)
b_mm = 95.0                           # track width [mm](mesure%20avec%20ton%20Thymio)
omega_scale = SPEED_TO_MMS / b_mm     # rad/s per (Thymio unit)
MIN_DT = 1e-3

def smooth_mavg(x, win=9):
    if len(x) < win or win < 3:
        return x.copy()
    y = np.convolve(x, np.ones(win)/win, mode="same")
    k = win//2
    y[:k] = x[:k]; y[-k:] = x[-k:]
    return y

async def set_motors(node, left, right):
    await node.set_variables({
        "motor.left.target": [int(left)],
        "motor.right.target": [int(right)],
    })

async def motion_profile(node, total_time):
    """
    Profil 'lignes droites + virages doux' pour exciter v et ω:
    - deux segments droits (vitesses moyenne/rapide),
    - virage gauche doux, virage droit doux,
    - répété jusqu’à total_time.
    """
    t0 = time.time()
    try:
        while time.time() - t0 < total_time:
            await set_motors(node, 10, 10); await asyncio.sleep(2.0)  # droit moyen
            await set_motors(node, 90, 90); await asyncio.sleep(2.0)  # droit rapide
            await set_motors(node, 10, 90); await asyncio.sleep(2.0)  # virage gauche
            await set_motors(node, 90, 10); await asyncio.sleep(2.0)  # virage droit
            await set_motors(node, 5, 5); await asyncio.sleep(2.0)  # droit lent
            await set_motors(node, 0, 0);     await asyncio.sleep(0.8)  # pause
    finally:
        await set_motors(node, 0, 0)

async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables({"motor.left.speed", "motor.right.speed"})

    vision = vu.Vision(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
                       MAP_WIDTH, MAP_HEIGHT, MATRIX_FILE_PATH,
                       thymio_id=THYMIO_MARKER_ID, goal_id=9999, min_area=100)

    try:
        # ===== Phase 1: STATIC =====
        print("Phase 1/3: Statique (ne pas bouger le robot)…")
        xs, ys, ths = [], [], []
        t0 = time.time()
        while time.time() - t0 < DUR_STATIC:
            frame = vision.get_warped_frame()
            if frame is None:
                await asyncio.sleep(Ts); continue
            pose = vision.get_thymio_pose(frame)
            if pose is not None:
                (px, py), ang_deg = pose
                xs.append(px); ys.append(py)
                ths.append(math.radians(ang_deg))
            cv2.imshow("Static", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            await asyncio.sleep(Ts)

        xs = np.array(xs, float); ys = np.array(ys, float); ths = np.array(ths, float)
        if xs.size < 40:
            print("Pas assez d’échantillons statiques.")
            return

        r_px_mm2 = float(np.var(xs, ddof=1) * (mm_per_px**2))
        r_py_mm2 = float(np.var(ys, ddof=1) * (mm_per_px**2))
        r_theta_rad2 = float(np.var(ths, ddof=1))

        print("\n[STATIC] R (bruits de mesure):")
        print(f"  r_p,x = {r_px_mm2:.3f} mm^2")
        print(f"  r_p,y = {r_py_mm2:.3f} mm^2")
        print(f"  r_theta = {r_theta_rad2:.3e} rad^2")

        # ===== Phase 2: MOTION profiling =====
        print("\nPhase 2/3: Mouvement (profil commandé)…")
        pxs, pys, ths_m, ts = [], [], [], []
        v_odom, w_odom, to = [], [], []

        motion_task = asyncio.create_task(motion_profile(node, DUR_MOTION))
        t0 = time.time()
        while time.time() - t0 < DUR_MOTION:
            frame = vision.get_warped_frame()
            if frame is None:
                await asyncio.sleep(Ts); continue

            pose = vision.get_thymio_pose(frame)
            now = time.time()
            if pose is not None:
                (px, py), ang_deg = pose
                pxs.append(px); pys.append(py)
                ths_m.append(math.radians(ang_deg)); ts.append(now)

                sL = float(node["motor.left.speed"])
                sR = float(node["motor.right.speed"])
                v_meas = 0.5*(sL + sR)*SPEED_TO_MMS     # mm/s
                w_meas = (sR - sL)*omega_scale          # rad/s
                v_odom.append(v_meas); w_odom.append(w_meas); to.append(now)

            cv2.imshow("Motion profiling", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            await asyncio.sleep(Ts)

        await motion_task
        await set_motors(node, 0, 0)

        pxs = np.array(pxs, float); pys = np.array(pys, float)
        ths_m = np.array(ths_m, float); ts = np.array(ts, float)
        v_odom = np.array(v_odom, float); w_odom = np.array(w_odom, float); to = np.array(to, float)

        if pxs.size < 50 or v_odom.size < 20:
            print("Pas assez d’échantillons mouvement.")
            return

        # v_cam (mm/s) via différentiation
        dt = np.diff(ts); mask = dt > MIN_DT
        vx_cam_px = np.diff(pxs)[mask] / dt[mask]
        vy_cam_px = np.diff(pys)[mask] / dt[mask]
        vx_cam = vx_cam_px * mm_per_px
        vy_cam = vy_cam_px * mm_per_px
        v_cam = np.sqrt(vx_cam**2 + vy_cam**2)

        # q_v: Var(Δv_cam)
        dv = np.diff(v_cam)
        q_v_mms2_per_step = float(np.var(dv, ddof=1)) if dv.size > 5 else 0.0

        # r_v,odom: split sur segment “steady” (milieu 60%)
        n_o = v_odom.size
        i0, i1 = int(0.2*n_o), int(0.8*n_o)
        v_seg = v_odom[i0:i1] if i1>i0 else v_odom
        var_vodom = float(np.var(v_seg, ddof=1)) if v_seg.size>5 else float(np.var(v_odom, ddof=1))
        alpha = 0.5
        r_v_odom_mms2 = max(1e-8, alpha*var_vodom)

        # q_theta: Var(Δe_theta) vs intégration ω
        if w_odom.size >= 2:
            w_i = np.interp(ts, to, w_odom)
            th_pred = np.zeros_like(ts); th_pred[0] = ths_m[0]
            for k in range(len(ts)-1):
                th_pred[k+1] = th_pred[k] + w_i[k]*(ts[k+1]-ts[k])
            th_meas = np.unwrap(ths_m); th_pred = np.unwrap(th_pred)
            e_th = th_meas - th_pred
            de = np.diff(e_th)
            q_theta_rad2_per_step = float(np.var(de, ddof=1)) if de.size>5 else 1e-4
        else:
            q_theta_rad2_per_step = 1e-4

        print("\n[MOTION] Q(process) et R(vitesse):")
        print(f"  q_v = {q_v_mms2_per_step:.3f} (mm/s)^2/step")
        print(f"  r_v,odom = {r_v_odom_mms2:.3f} (mm/s)^2")
        print(f"  q_theta = {q_theta_rad2_per_step:.3e} rad^2/step")

        # ===== Phase 3: PREDICTION ONLY (q_x, q_y) =====
        print("\nPhase 3/3: Prédiction pure (profil commandé) pour q_x, q_y…")
        xs_cam, ys_cam, ts2 = [], [], []
        xs_pred, ys_pred = [], []
        xhat = yhat = None
        tprev = None

        motion_task2 = asyncio.create_task(motion_profile(node, DUR_QPOS))
        t0 = time.time()
        while time.time() - t0 < DUR_QPOS:
            frame = vision.get_warped_frame()
            if frame is None:
                await asyncio.sleep(Ts); continue
            pose = vision.get_thymio_pose(frame)
            now = time.time()
            if pose is not None:
                (px, py), ang_deg = pose
                x_cam = px * mm_per_px
                y_cam = py * mm_per_px
                theta = math.radians(ang_deg)

                if xhat is None:
                    xhat, yhat = x_cam, y_cam
                    tprev = now
                else:
                    dtp = now - tprev
                    if dtp > MIN_DT:
                        sL = float(node["motor.left.speed"])
                        sR = float(node["motor.right.speed"])
                        v_meas = 0.5*(sL + sR)*SPEED_TO_MMS
                        xhat = xhat + v_meas*math.cos(theta)*dtp
                        yhat = yhat + v_meas*math.sin(theta)*dtp
                        tprev = now

                xs_cam.append(x_cam); ys_cam.append(y_cam); ts2.append(now)
                xs_pred.append(xhat); ys_pred.append(yhat)

            cv2.imshow("Prediction only (q_x,q_y)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            await asyncio.sleep(Ts)

        await motion_task2
        await set_motors(node, 0, 0)

        xs_cam = np.array(xs_cam); ys_cam = np.array(ys_cam)
        xs_pred = np.array(xs_pred); ys_pred = np.array(ys_pred)
        if xs_cam.size < 50:
            print("Pas assez d’échantillons pour q_x,q_y.")
            return

        ex = xs_cam - xs_pred
        ey = ys_cam - ys_pred
        exc = ex - np.mean(ex); eyc = ey - np.mean(ey)
        vx = smooth_mavg(exc**2, win=7)
        vy = smooth_mavg(eyc**2, win=7)
        k = np.arange(len(vx))
        Ax = np.vstack([k, np.ones_like(k)]).T
        slope_x, _ = np.linalg.lstsq(Ax, vx, rcond=None)[0]
        slope_y, _ = np.linalg.lstsq(Ax, vy, rcond=None)[0]
        qx_mm2_per_step = float(max(slope_x, 1e-3))
        qy_mm2_per_step = float(max(slope_y, 1e-3))

        print("\n[PRED ONLY] Q(process) position:")
        print(f"  q_x ≈ {qx_mm2_per_step:.3f} mm^2/step")
        print(f"  q_y ≈ {qy_mm2_per_step:.3f} mm^2/step")

        # ===== Récapitulatif final =====
        print("\n=== Récapitulatif à mettre dans le filtre ===")
        print(f"R_pos = diag({r_px_mm2:.3f}, {r_py_mm2:.3f})   [mm^2]")
        print(f"R_theta = {r_theta_rad2:.3e}                 [rad^2]")
        print(f"R_v(odom) = {r_v_odom_mms2:.3f}             [(mm/s)^2]")
        print(f"Q = diag(q_x, q_y, q_theta, q_v) = diag({qx_mm2_per_step:.3f}, {qy_mm2_per_step:.3f}, {q_theta_rad2_per_step:.3e}, {q_v_mms2_per_step:.3f})")

        print("\nTips tuning (Ex. 8):")
        print("- Si la pose ‘saute’ aux updates caméra: ↑ r_p (ou r_theta) ou ↓ q_x,q_y/q_theta.")
        print("- Si la pose ‘rame’: ↓ r_p (ou r_theta) ou ↑ q_x,q_y/q_theta.")
        print("- Si v est trop ‘raide’ en virage: ↑ q_v. Si v trop bruitée: ↑ r_v,odom.")
        print("- Vérifie innovations vs S = H P̄ H^T + R (cf. Ex. 8).")

    finally:
        await node.unlock()
        vision.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
import numpy as np

class EKF2D:
    # State x = [px, py, vx, vy] in pixels and px/s
    def __init__(self, Ts, x0=None, P0=None,
                 q_pos=0.2, q_vel=8.0,
                 r_cam_posx=6.0, r_cam_posy=6.0,r_vel_cam_derived=120.0):
        self.Ts = Ts
        self.x = np.zeros(4) if x0 is None else np.array(x0, dtype=float)
        self.P = np.diag([1e3, 1e3, 1e2, 1e2]) if P0 is None else np.array(P0, dtype=float)

        # Constant-velocity model
        self.A = np.array([[1, 0, Ts, 0],
                           [0, 1, 0,  Ts],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurements
        #H_cam_pos selects position states
        self.H_cam_pos = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])
        self.R_cam_pos = np.diag([r_cam_posx, r_cam_posy])

        #H_vel slects velocity states 
        self.H_vel = np.array([[0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.R_vel_cam = np.diag([r_vel_cam_derived, r_vel_cam_derived])  # velocity derived from cam deltas

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def _update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def update_cam_pos(self, px, py):
        self._update(np.array([px, py]), self.H_cam_pos, self.R_cam_pos)

    def update_vel_cam(self, vx, vy):
        self._update(np.array([vx, vy]), self.H_vel, self.R_vel_cam)

    def get_state(self):
        return self.x.copy(), self.P.copy()
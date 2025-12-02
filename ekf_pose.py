import numpy as np

class EKFPose:
    # x = [x_mm, y_mm, theta_rad, v_mm_s]
    def __init__(self, Ts,
                 x0=None, P0=None,
                 # Q (process): mm^2, rad^2, (mm/s)^2 per step
                 q_x=5.0, q_y=5.0, q_theta=1e-3, q_v=200.0,
                 # R (measure): mm^2, rad^2, (mm/s)^2
                 r_pos_x=400.0, r_pos_y=400.0,
                 r_theta=(np.deg2rad(2.0))**2,
                 r_v=50.0**2):
        self.Ts = Ts
        self.x = np.array([0.0, 0.0, 0.0, 0.0]) if x0 is None else np.array(x0, float)
        self.P = np.diag([1e4, 1e4, 1.0, 1e3]) if P0 is None else np.array(P0, float)
        self.Q = np.diag([q_x, q_y, q_theta, q_v])
        self.R_pos = np.diag([r_pos_x, r_pos_y])
        self.R_theta = np.array([[r_theta]])
        self.R_v = np.array([[r_v]])

    # g(u,x): u = [v_meas (mm/s), omega_meas (rad/s)]
    def f(self, x, u):
        Ts = self.Ts
        px, py, th, v = x
        v_meas = float(u[0])
        omega  = float(u[1])
        pxn = px + v_meas * np.cos(th) * Ts
        pyn = py + v_meas * np.sin(th) * Ts
        thn = th + omega * Ts
        vn  = v_meas
        return np.array([pxn, pyn, thn, vn])

    # Jacobien G = ∂g/∂x autour de (x,u)
    def G_jacobian(self, x, u):
        Ts = self.Ts
        th= x[2]
        v_meas = float(u[0])
        c, s = np.cos(th), np.sin(th)
        G = np.array([
            [1.0, 0.0, -v_meas*s*Ts,  0.0],
            [0.0, 1.0,  v_meas*c*Ts,  0.0],
            [0.0, 0.0,  1.0,          0.0],
            [0.0, 0.0,  0.0,          0.0]  # v' ne dépend pas de v (v' = v_meas)
        ])
        return G

    # Mesures
    def h_pos(self, x):     return x[:2]
    def H_pos(self, x):     return np.array([[1,0,0,0],[0,1,0,0]])

    def h_theta(self, x):   return np.array([x[2]])
    def H_theta(self, x):   return np.array([[0,0,1,0]])

    def h_v(self, x):       return np.array([x[3]])
    def H_v(self, x):       return np.array([[0,0,0,1]])

    # EKF steps
    def predict(self, u):
        xbar = self.f(self.x, u)
        G = self.G_jacobian(self.x, u)
        Pbar = G @ self.P @ G.T + self.Q
        self.x, self.P = xbar, Pbar

    def update(self, z, h_fun, H_fun, R):
        zhat = h_fun(self.x)
        H = H_fun(self.x)
        y = z - zhat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    # Wrappers
    def update_pos(self, x_mm, y_mm):
        self.update(np.array([x_mm, y_mm], float), self.h_pos, self.H_pos, self.R_pos)
    def update_theta(self, theta_rad):
        self.update(np.array([theta_rad], float), self.h_theta, self.H_theta, self.R_theta)
    def update_v(self, v_mm_s):
        self.update(np.array([v_mm_s], float), self.h_v, self.H_v, self.R_v)

    def get_state(self):
        return self.x.copy(), self.P.copy()
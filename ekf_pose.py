import cv2
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

Q_X=0.25
Q_Y=0.25
Q_TH=2.752e-03
R_POSX=0.00001
R_POSY=0.00001
R_TH=2.229e-06


class EKFPose:
    # x = [x_mm, y_mm, theta_rad, v_mm_s]
    def __init__(self,
                 x0=None, P0=None,
                 # Q (process): mm^2, rad^2
                 q_x=Q_X, q_y=Q_Y, q_theta=Q_TH,
                 # R (measure): mm^2, rad^2
                 r_pos_x=R_POSX, r_pos_y=R_POSY,
                 r_theta=R_TH,
                 save_data = False
                ):
        self.x = np.array([0.0, 0.0, 0.0]) if x0 is None else np.array(x0, float)
        self.P = np.diag([1e4, 1e4, 1.0]) if P0 is None else np.array(P0, float)
        self.Q = np.diag([q_x, q_y, q_theta])
        self.R_pos = np.diag([r_pos_x, r_pos_y])
        self.R_theta = np.array([[r_theta]])
        self.start_time = perf_counter()
        self.last_measurement = perf_counter()
        self.save_data = save_data
        if save_data:
            self.x_predicted = []
            self.x_variance = []
            self.data_time = []

    # g(u,x): u = [v_meas (mm/s), omega_meas (rad/s)]
    def f(self, x, u, delta_t):
        # delta_t = perf_counter() - self.last_measurement
        # self.last_measurement = perf_counter()
        px, py, th = x
        v_meas = float(u[0])
        omega  = float(u[1])
        pxn = px + v_meas * np.sin(th) * delta_t
        pyn = py + v_meas * -np.cos(th) * delta_t
        thn = th + omega * delta_t
        return np.array([pxn, pyn, thn])

    # Jacobien G = ∂g/∂x autour de (x,u)
    def G_jacobian(self, x, u, delta_t):
        #delta_t = perf_counter() - self.last_measurement
        th= x[2]
        v_meas = float(u[0])
        c, s = np.cos(th), np.sin(th)
        G = np.array([
            [1.0, 0.0, -v_meas*s*delta_t],
            [0.0, 1.0,  v_meas*c*delta_t],
            [0.0, 0.0,  1.0],
        ])
        return G

    # Mesures
    def h_pos(self, x):     return x[:2]
    def H_pos(self, x):     return np.array([[1,0,0],[0,1,0]])

    def h_theta(self, x):   return np.array([x[2]])
    def H_theta(self, x):   return np.array([[0,0,1]])

    # EKF steps
    def predict(self, u):
        delta_t = perf_counter() - self.last_measurement
        self.last_measurement = perf_counter()
        xbar = self.f(self.x, u, delta_t)
        G = self.G_jacobian(self.x, u, delta_t)
        Pbar = G @ self.P @ G.T + self.Q
        self.x, self.P = xbar, Pbar
        if self.save_data:
            self.x_predicted.append(self.x[0])
            self.x_variance.append(self.P[0,0])
            self.data_time.append(self.last_measurement - self.start_time)


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

    def get_state(self):
        return self.x.copy(), self.P.copy()

    # Data saving
    def stop_save(self):
        time = np.array(self.data_time)[20:]
        predicted = np.array(self.x_predicted)[20:]
        variance  = np.array(self.x_variance)[20:]
        plt.plot(time, predicted, 'b-', label='x_odometric')
        plt.fill_between(time, predicted - variance, predicted + variance, color='blue', alpha=0.2)
        plt.show()

    # Kalman Visualization
    def draw_covariance_ellipse(self, frame, mm_per_px, scale_factor=3.0):
        """
        Draws the XY uncertainty ellipse and Theta uncertainty wedge.
        """
        P = self.P.copy()

        # Multiplier to make small errors visible to the human eye
        VISUAL_GAIN = 10.0 
        
        # Calculate Angle of the XY error
        angle = np.degrees(np.arctan2(P[1,1], P[0, 0]))
        
        # Calculate Axis Lengths (3-sigma * visual_gain)
        # We apply VISUAL_GAIN here to make it huge enough to see
        width = 2 * scale_factor * np.sqrt(P[0,0]) / mm_per_px * VISUAL_GAIN
        height = 2 * scale_factor * np.sqrt(P[1,1]) / mm_per_px * VISUAL_GAIN
        
        # Clamp minimum size so it doesn't vanish
        width = max(20, width) 
        height = max(20, height)
        
        # Fixed Position (Bottom Left)
        fixed_center = (200, frame.shape[0] - 200)
        
        try:
            # --- DRAW XY ELLIPSE (Blue) ---
            # We draw a filled semi-transparent ellipse if possible, but OpenCV
            cv2.ellipse(frame, fixed_center, (int(width/2), int(height/2)), 
                        int(angle), 0, 360, (255, 100, 0), -1)
            
            # --- DRAW THETA UNCERTAINTY (Yellow Wedge) ---
            var_theta = P[2, 2]
            if var_theta < 0: var_theta = 0
            std_theta = np.sqrt(var_theta)
            
            # Calculate angular spread (3-sigma)
            spread_deg = np.degrees(3 * std_theta)
            
            # Limit spread for visualization sanity
            spread_deg = min(spread_deg, 180)
            
            # Arrow Length (proportional to ellipse size or fixed)
            arrow_len = 100
            
            # Draw the "Cone" of uncertainty
            # We assume "Up" (-90 degrees in OpenCV) is the reference direction
            start_angle = -90 - spread_deg
            end_angle = -90 + spread_deg
            
            # Draw filled arc section
            cv2.ellipse(frame, fixed_center, (arrow_len, arrow_len), 0, start_angle, end_angle, (0, 200, 255), -1)
            
            # Draw the "Mean" Arrow (Center of the cone)
            p1 = fixed_center
            p2 = (fixed_center[0], fixed_center[1] - arrow_len)
            cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.2)

            # --- LABELS ---
            cv2.putText(frame, f"EKF Uncertainty (x{int(VISUAL_GAIN)})", (fixed_center[0]-100, fixed_center[1]+120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, f"Theta: +/- {spread_deg:.1f} deg", (fixed_center[0]-80, fixed_center[1]+145), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        except Exception as e:
            print(f"Viz Error: {e}")
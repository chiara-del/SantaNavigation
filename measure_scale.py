import cv2
import numpy as np
import math
import vision_utils as vu

# --- CONFIGURATION ---
CAMERA_INDEX = 1   # Set this to your USB camera index (e.g., 1)
MAP_WIDTH = 700
MAP_HEIGHT = 500
MATRIX_FILE_PATH = "my_matrix.npy"

# --- GLOBAL VARIABLES ---
points = []
current_frame = None
scale_ratio = 0.0 # pixels per cm

def click_event(event, x, y, flags, params):
    """Handles mouse clicks to measure distance."""
    global points, current_frame, scale_ratio
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        # Draw the point
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Measure Scale", current_frame)
        
        if len(points) == 2:
            # Calculate pixel distance
            p1 = points[0]
            p2 = points[1]
            dist_px = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Draw the line
            cv2.line(current_frame, p1, p2, (255, 0, 0), 2)
            cv2.imshow("Measure Scale", current_frame)
            
            print(f"\nPixel Distance: {dist_px:.2f} pixels")
            
            # Ask user for real world distance
            try:
                real_dist = float(input("Enter the REAL distance in cm between these points: "))
                if real_dist > 0:
                    scale_ratio = dist_px / real_dist
                    print(f"------------------------------------------------")
                    print(f"✅ SCALE: {scale_ratio:.2f} pixels per cm")
                    print(f"------------------------------------------------")
                    
                    # Calculate Thymio radius example
                    thymio_radius_cm = 5.5 # Approx radius
                    thymio_radius_px = int(thymio_radius_cm * scale_ratio)
                    print(f"Example: Thymio Radius ({thymio_radius_cm} cm) = {thymio_radius_px} pixels")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            
            # Reset for next measurement
            points = []

def main():
    global current_frame
    
    # Setup
    cap = vu.setup_camera(CAMERA_INDEX, 1920, 1080)
    if cap is None: return
    
    matrix = vu.load_transform_matrix(MATRIX_FILE_PATH)
    if matrix is None:
        print("Error: Matrix not found. Run main.py to calibrate first.")
        return

    cv2.namedWindow("Measure Scale")
    cv2.setMouseCallback("Measure Scale", click_event)
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Click on a point on the map.")
    print("2. Click on a second point.")
    print("3. Look at the terminal and enter the real distance in cm.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Get Top-Down Map
        top_down_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))
        
        # We only update the display if we aren't in the middle of clicking
        if len(points) == 0:
            current_frame = top_down_map.copy()
            cv2.imshow("Measure Scale", current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
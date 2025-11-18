import cv2
import numpy as np

# --- 1. CONFIGURATION ---
USE_WEBCAM = True  # Set to True to use live webcam, False for static image
CAMERA_INDEX = 1    # Your camera index
IMAGE_PATH = "my_arena_image.jpg" # Path to your test image

# A "dummy" function that does nothing.
# createTrackbar requires a callback function, but we don't need one.
def nothing(x):
    pass

# --- 2. SETUP ---
# Create a window to hold the trackbars
cv2.namedWindow("HSV Tuner")

# Create 6 trackbars for the HSV range
# Arguments: (Trackbar Name, Window Name, Min Value, Max Value, Callback Function)
cv2.createTrackbar("H_min", "HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("S_min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("V_min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("H_max", "HSV Tuner", 179, 179, nothing) # Start H_max at 179
cv2.createTrackbar("S_max", "HSV Tuner", 255, 255, nothing) # Start S_max at 255
cv2.createTrackbar("V_max", "HSV Tuner", 255, 255, nothing) # Start V_max at 255

# Set initial values (optional, but good for finding white)
# You can comment these out to start from 0
cv2.setTrackbarPos("V_min", "HSV Tuner", 100) # Good start for V_min
cv2.setTrackbarPos("S_max", "HSV Tuner", 25)  # Good start for S_max (for white)

# Setup camera or load image
if USE_WEBCAM:
    cap = cv2.VideoCapture(CAMERA_INDEX)
else:
    # Load the image once
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        exit()
    # Resize for consistent display if it's too big
    image = cv2.resize(image, (640, 480))

print("Starting HSV Tuner...")
print("Drag the sliders to find the perfect values.")
print("Look at the 'Mask' window to see the result.")
print("Press 'q' to quit and print the final values.")

# --- 3. MAIN LOOP ---
while True:
    # --- Get Frame ---
    if USE_WEBCAM:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        frame = cv2.resize(frame, (640, 480)) # Resize for display
    else:
        frame = image.copy() # Use the same image every loop

    # --- Get Slider Positions ---
    h_min = cv2.getTrackbarPos("H_min", "HSV Tuner")
    s_min = cv2.getTrackbarPos("S_min", "HSV Tuner")
    v_min = cv2.getTrackbarPos("V_min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("H_max", "HSV Tuner")
    s_max = cv2.getTrackbarPos("S_max", "HSV Tuner")
    v_max = cv2.getTrackbarPos("V_max", "HSV Tuner")

    # --- Create Bounds ---
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # --- Apply Mask ---
    # 1. Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 2. Create the mask
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    # 3. (Optional) Apply mask to original image to see what's being kept
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # --- Display ---
    cv2.imshow("Original Image", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result (Masked)", result)

    # --- Quit ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
if USE_WEBCAM:
    cap.release()
cv2.destroyAllWindows()

# --- 5. PRINT RESULTS ---
print("\n--- Final Tuned Values ---")
print(f"LOWER_WHITE_HSV = np.array([{h_min}, {s_min}, {v_min}])")
print(f"UPPER_WHITE_HSV = np.array([{h_max}, {s_max}, {v_max}])")
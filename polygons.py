import cv2
import numpy as np

def mask_to_polygons(
    mask,
    blur_ksize=5,
    morph_kernel_size=5,
    morph_iters=2,
    epsilon_factor=0.02,
    min_area=100
):
    """
    Convert a binary mask (white shapes on black) into simplified polygons.

    Parameters
    ----------
    mask : np.ndarray
        Binary image (uint8). Shapes should be white (255), background black (0).
        Can also be grayscale; we'll threshold it.
    blur_ksize : int
        Kernel size for Gaussian blur (must be odd). 0 or 1 to disable.
    morph_kernel_size : int
        Kernel size for morphological operations (closing + opening).
    morph_iters : int
        Number of iterations for morphology.
    epsilon_factor : float
        Simplification factor for approxPolyDP. Higher = fewer vertices.
        Typical range: 0.01–0.05.
    min_area : float
        Minimum contour area to keep (to filter tiny noise).

    Returns
    -------
    polygons : list of np.ndarray
        Each element is an array of shape (N, 2) with (x, y) vertex coordinates.
        Example: [array([[x1, y1], [x2, y2], ...]), ...]
    """

    # 1) Ensure single channel
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    # 2) Optional blur to smooth noise before threshold
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        mask_gray = cv2.GaussianBlur(mask_gray, (blur_ksize, blur_ksize), 0)

    # 3) Binarize (in case it's not strictly 0/255 already)
    _, th = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4) Morphological operations to fill gaps and remove specks
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 5) Find contours on cleaned mask
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # ignore tiny blobs

        # Optional: uncomment next line if you want convex obstacles only
        # cnt = cv2.convexHull(cnt)

        # 6) Simplify contour -> polygon
        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # shape: (N, 1, 2)

        # Reshape to (N, 2) and store as int
        poly = approx.reshape(-1, 2)
        polygons.append(poly)

    return polygons
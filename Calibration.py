import cv2
import numpy as np
import os

# Calibrate
def calibrate():

    pattern = (7, 7)
    square_size = 25.0

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    image_folder = "Calibration"

    for filename in os.listdir(image_folder):

        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, pattern)

        if not ret:
            print(f"Not found: {filename}")
            continue

        imgpoints.append(corners.astype(np.float32))
        objpoints.append(objp.copy())

        # # -------- DISPLAY CHECK --------
        # vis = img.copy()
        # cv2.drawChessboardCorners(vis, pattern, corners, ret)
        #
        # cv2.imshow("Checkerboard Detection", vis)
        # cv2.waitKey(500)   # show for 0.5 seconds
        # # --------------------------------

        print(f"Detected: {filename}")

    cv2.destroyAllWindows()

    _, K, _, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (1280, 1280),
        None,
        None
    )

    return K
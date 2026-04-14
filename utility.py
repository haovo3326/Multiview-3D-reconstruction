import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch

def R_to_quaternion(R_mat):
    r = R.from_matrix(R_mat)
    q = r.as_quat()  # [x, y, z, w]
    # convert to [w, x, y, z]
    return np.array([q[3], q[0], q[1], q[2]])

def quaternion_to_R(q):
    # input q: [w, x, y, z]
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)

    # SciPy expects [x, y, z, w]
    q_scipy = np.array([q_norm[1], q_norm[2], q_norm[3], q_norm[0]])

    R_mat = R.from_quat(q_scipy).as_matrix()
    return R_mat

def build_projection_matrix(K, R, t):
    return (K @ np.hstack((R, t))).astype(np.float64)

def triangulate_points(Pa, Pb, pts_a, pts_b):
    X_h = cv2.triangulatePoints(Pa, Pb, pts_a.T, pts_b.T)
    X = (X_h[:3] / X_h[3]).T
    return X.astype(np.float64)

def normalize_brightness(img):
    # img: torch tensor [3,H,W] in [0,1]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # CLAHE (best choice)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # back to 3-channel
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    gray = gray.astype(np.float32) / 255.0
    gray = torch.from_numpy(gray).permute(2, 0, 1)

    return gray

def save_point_cloud_ply(points, filename="point_cloud.ply"):
    points = np.asarray(points, dtype=np.float64)

    # filter valid points
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    if points.shape[0] == 0:
        print("No valid points to save.")
        return

    with open(filename, "w") as f:
        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # write points
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"Saved {points.shape[0]} points to {filename}")
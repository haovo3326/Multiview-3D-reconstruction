import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch
import json
from pathlib import Path

def extract_intrinsics(K):
    fx = K[0, 0]
    fy = K[1, 1]
    s = K[0, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return np.array([fx, fy, cx, cy, s])

def clone_intrinsics(k):
    return k.copy()

def extract_radial_distortion(dist_coeffs):
    if dist_coeffs is None:
        return np.zeros(3, dtype=np.float64)

    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    radial = np.zeros(3, dtype=np.float64)

    if dist.size > 0:
        radial[0] = dist[0]
    if dist.size > 1:
        radial[1] = dist[1]
    if dist.size > 4:
        radial[2] = dist[4]

    return radial

def clone_radial_distortion(d):
    return np.asarray(d, dtype=np.float64).reshape(3).copy()

def build_radial_dist_coeffs(d):
    k1, k2, k3 = np.asarray(d, dtype=np.float64).reshape(3)
    return np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)

def build_K_from_intrinsics(k):
    fx, fy, cx, cy, s = np.asarray(k, dtype=np.float64).reshape(5)
    return np.array([
        [fx, s, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

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

def project_point_with_radial(k, d, q, t, point3d):
    fx, fy, cx, cy, s = np.asarray(k, dtype=np.float64).reshape(5)
    k1, k2, k3 = np.asarray(d, dtype=np.float64).reshape(3)
    R_mat = quaternion_to_R(q)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    X = np.asarray(point3d, dtype=np.float64).reshape(3)

    Z = R_mat @ X + t
    if abs(Z[2]) < 1e-12:
        return None

    xn = Z[0] / Z[2]
    yn = Z[1] / Z[2]
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    xd = xn * radial
    yd = yn * radial

    return np.array([
        fx * xd + s * yd + cx,
        fy * yd + cy
    ], dtype=np.float64)

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

def save_point_cloud_ply(points, filename="point_cloud.ply", colors=None):
    points = np.asarray(points, dtype=np.float64)
    colors = None if colors is None else np.asarray(colors, dtype=np.float64)

    # filter valid points
    mask = np.isfinite(points).all(axis=1)
    if colors is not None:
        mask &= np.isfinite(colors).all(axis=1)

    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if points.shape[0] == 0:
        print("No valid points to save.")
        return

    if colors is not None:
        colors = np.clip(colors, 0.0, 1.0)
        colors = np.rint(colors * 255.0).astype(np.uint8)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # write points
        for i, p in enumerate(points):
            if colors is None:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
            else:
                c = colors[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"Saved {points.shape[0]} points to {path}")

def save_camera_info(camera_matrices, filename="save/cameras.json"):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    cameras = []
    for camera_id, cam in enumerate(camera_matrices):
        k = np.asarray(cam["k"], dtype=np.float64).reshape(5)
        d = np.asarray(cam["d"], dtype=np.float64).reshape(3)
        q = np.asarray(cam["q"], dtype=np.float64).reshape(4)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

        cameras.append({
            "camera_id": camera_id,
            "intrinsic_parameters": {
                "fx": float(k[0]),
                "fy": float(k[1]),
                "cx": float(k[2]),
                "cy": float(k[3]),
                "skew": float(k[4])
            },
            "intrinsic_matrix": build_K_from_intrinsics(k).tolist(),
            "radial_distortion": {
                "k1": float(d[0]),
                "k2": float(d[1]),
                "k3": float(d[2])
            },
            "quaternion_wxyz": q.tolist(),
            "translation": t.tolist()
        })

    payload = {
        "camera_count": len(cameras),
        "cameras": cameras
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(cameras)} cameras to {path}")

def load_camera_info(filename="save/cameras.json"):
    path = Path(filename)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    camera_matrices = []
    for cam in payload["cameras"]:
        intrinsic = cam["intrinsic_parameters"]
        radial = cam["radial_distortion"]

        camera_matrices.append({
            "k": np.array([
                intrinsic["fx"],
                intrinsic["fy"],
                intrinsic["cx"],
                intrinsic["cy"],
                intrinsic["skew"]
            ], dtype=np.float64),
            "d": np.array([
                radial["k1"],
                radial["k2"],
                radial["k3"]
            ], dtype=np.float64),
            "q": np.asarray(cam["quaternion_wxyz"], dtype=np.float64).reshape(4),
            "t": np.asarray(cam["translation"], dtype=np.float64).reshape(3)
        })

    return camera_matrices

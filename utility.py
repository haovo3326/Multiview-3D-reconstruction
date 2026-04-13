import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def R_to_quaternion(R_mat):
    r = R.from_matrix(R_mat)
    q = r.as_quat()  # [x, y, z, w]
    # convert to [w, x, y, z]
    return np.array([q[3], q[0], q[1], q[2]])

def quaternion_to_R(q):
    # input q: [w, x, y, z]
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)

    # SciPy expects [x, y, z, w]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])

    R_mat = R.from_quat(q_scipy).as_matrix()
    return R_mat

def build_projection_matrix(K, R, t):
    return (K @ np.hstack((R, t))).astype(np.float64)

def triangulate_points(Pa, Pb, pts_a, pts_b):
    X_h = cv2.triangulatePoints(Pa, Pb, pts_a.T, pts_b.T)
    X = (X_h[:3] / X_h[3]).T
    return X.astype(np.float64)
import Calibration
import DSU
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt



device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)


def keypoints_and_descriptors(image_path):
    image = load_image(image_path).to(device)
    feats = extractor.extract(image)
    keypoints = feats['keypoints'][0].cpu().numpy().astype(np.float32)
    return feats, keypoints


def matches_idx(feats1, feats2):
    matches = matcher({'image0': feats1, 'image1': feats2})
    matches = rbd(matches)
    return matches['matches'].cpu().numpy()


def pose_matching(kpts1, kpts2, idx12, K):
    pts1 = kpts1[idx12[:, 0]]
    pts2 = kpts2[idx12[:, 1]]

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    mask = mask.ravel().astype(bool)
    idx12 = idx12[mask]
    pts1 = pts1[mask]
    pts2 = pts2[mask]

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    pose_mask = pose_mask.ravel().astype(bool)
    idx12 = idx12[pose_mask]

    return idx12, R, t


def build_projection_matrix(K, R, t):
    return (K @ np.hstack((R, t))).astype(np.float64)

def save_ply(points, filename="output.ply"):
    points = points[np.isfinite(points).all(axis=1)]

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def triangulate_points(Pa, Pb, pts_a, pts_b):
    X_h = cv2.triangulatePoints(Pa, Pb, pts_a.T, pts_b.T)
    X = (X_h[:3] / X_h[3]).T
    return X.astype(np.float64)


def remap_track_points(dsu, track_to_point):
    new_map = {}
    for old_root, X in track_to_point.items():
        new_root = dsu.find(old_root)
        if new_root not in new_map:
            new_map[new_root] = X
    return new_map


def assign_triangulated_points(dsu, idx_ab, Xa, img_a_id):
    for j, pair in enumerate(idx_ab):
        i_a = int(pair[0])
        root = dsu.find((img_a_id, i_a))
        if root not in track_to_point:
            track_to_point[root] = Xa[j]


def pnp_from_tracks(dsu, track_to_point, idx_prev_curr, kpts_curr, prev_img_id, curr_img_id, K):
    obj_points = []
    img_points = []

    for i_prev, i_curr in idx_prev_curr:
        node_prev = (prev_img_id, int(i_prev))
        if node_prev not in dsu.parent:
            continue

        root = dsu.find(node_prev)
        if root not in track_to_point:
            continue

        obj_points.append(track_to_point[root])
        img_points.append(kpts_curr[int(i_curr)])

    obj_points = np.asarray(obj_points, dtype=np.float32)
    img_points = np.asarray(img_points, dtype=np.float32)

    if len(obj_points) < 6:
        raise ValueError(f"Not enough 2D-3D correspondences for PnP on image {curr_img_id}: {len(obj_points)}")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=8.0,
        confidence=0.999,
        iterationsCount=1000
    )

    if not ok:
        raise ValueError(f"PnP failed for image {curr_img_id}")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    return R.astype(np.float64), t.astype(np.float64), obj_points, img_points, inliers


# ---------- load ----------
K = Calibration.calibrate().astype(np.float64)

feats1, kpts1 = keypoints_and_descriptors('Sample/Image 1.png')
feats2, kpts2 = keypoints_and_descriptors('Sample/Image 2.png')
feats3, kpts3 = keypoints_and_descriptors('Sample/Image 3.png')
feats4, kpts4 = keypoints_and_descriptors('Sample/Image 4.png')

idx12 = matches_idx(feats1, feats2)
idx23 = matches_idx(feats2, feats3)
idx34 = matches_idx(feats3, feats4)

dsu = DSU.DSU()
track_to_point = {}

# ---------- image 1-2 ----------
idx12, R12, t12 = pose_matching(kpts1, kpts2, idx12, K)

for i1, i2 in idx12:
    dsu.union((1, int(i1)), (2, int(i2)))

P1 = build_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
P2 = build_projection_matrix(K, R12, t12)

pts1_12 = kpts1[idx12[:, 0]]
pts2_12 = kpts2[idx12[:, 1]]
X12 = triangulate_points(P1, P2, pts1_12, pts2_12)

for j, (i1, i2) in enumerate(idx12):
    root = dsu.find((1, int(i1)))
    track_to_point[root] = X12[j]

# ---------- image 3 via PnP ----------
R3, t3, obj3, img3, inliers3 = pnp_from_tracks(
    dsu, track_to_point, idx23, kpts3, 2, 3, K
)
P3 = build_projection_matrix(K, R3, t3)

# now merge 2-3 tracks
for i2, i3 in idx23:
    dsu.union((2, int(i2)), (3, int(i3)))

track_to_point = remap_track_points(dsu, track_to_point)

# triangulate new tracks from 2-3 that do not already have a 3D point
pts2_23 = kpts2[idx23[:, 0]]
pts3_23 = kpts3[idx23[:, 1]]
X23 = triangulate_points(P2, P3, pts2_23, pts3_23)

for j, (i2, i3) in enumerate(idx23):
    root = dsu.find((2, int(i2)))
    if root not in track_to_point:
        track_to_point[root] = X23[j]

# ---------- image 4 via PnP ----------
R4, t4, obj4, img4, inliers4 = pnp_from_tracks(
    dsu, track_to_point, idx34, kpts4, 3, 4, K
)
P4 = build_projection_matrix(K, R4, t4)

# now merge 3-4 tracks
for i3, i4 in idx34:
    dsu.union((3, int(i3)), (4, int(i4)))

track_to_point = remap_track_points(dsu, track_to_point)

# triangulate new tracks from 3-4 that do not already have a 3D point
pts3_34 = kpts3[idx34[:, 0]]
pts4_34 = kpts4[idx34[:, 1]]
X34 = triangulate_points(P3, P4, pts3_34, pts4_34)

for j, (i3, i4) in enumerate(idx34):
    root = dsu.find((3, int(i3)))
    if root not in track_to_point:
        track_to_point[root] = X34[j]

# ---------- output ----------
tracks = dsu.groups()

# Collect all 3D points
points = np.array(list(track_to_point.values()))
save_ply(points, "pointcloud.ply")

# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
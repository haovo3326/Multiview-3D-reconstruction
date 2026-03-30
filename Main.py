import os

import numpy as np

import Utility
import cv2
import DSU
import Calibration
import matplotlib.pyplot as plt

def feature_matching(K, img1, img2):
    pts1, pts2 = Utility.get_correspondence(img1, img2)
    R, t, pts1_unique, pts2_unique = Utility.rounding_and_unique(K, pts1, pts2)
    return R, t, zip(pts1_unique, pts2_unique)

root = os.getcwd()
image1_path = os.path.join(root, "Sample", "Image 1.png")
image2_path = os.path.join(root, "Sample", "Image 2.png")
image3_path = os.path.join(root, "Sample", "Image 3.png")
image4_path = os.path.join(root, "Sample", "Image 4.png")

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)
image4 = cv2.imread(image4_path)

image1 = cv2.resize(image1, (860, 860))
image2 = cv2.resize(image2, (860, 860))
image3 = cv2.resize(image3, (860, 860))
image4 = cv2.resize(image4, (860, 860))


K = Calibration.calibrate()

R12, t12, matches12 = feature_matching(K, image1, image2)
R23, t23, matches23 = feature_matching(K, image2, image3)
R34, t34, matches34 = feature_matching(K, image3, image4)

dsu = DSU.DSU()

for pt1, pt2 in matches12:
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    dsu.union((1, p1), (2, p2))

for pt1, pt2 in matches23:
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    dsu.union((2, p1), (3, p2))

for pt1, pt2 in matches34:
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    dsu.union((3, p1), (4, p2))

tracks = dsu.groups()
length = [0, 0, 0, 0, 0]
for track in tracks:
    length[len(track)] += 1
print(length)

# Solving camera matrices
R1 = np.eye(3)
t1 = np.zeros((3, 1))

R2 = R12
t2 = t12

R3 = R23 @ R2
t3 = R23 @ t2 + t23

R4 = R34 @ R3
t4 = R34 @ t3 + t34

P1 = K @ np.hstack((R1, t1))
P2 = K @ np.hstack((R2, t2))
P3 = K @ np.hstack((R3, t3))
P4 = K @ np.hstack((R4, t4))

camera = [P1, P2, P3, P4]

points3D = []

for track in tracks:
    if len(track) > 2:
        print(track)
    A = []
    for p in track:
        camera_id, coord = p
        u, v = coord
        P = camera[camera_id - 1]
        p1 = P[0]
        p2 = P[1]
        p3 = P[2]
        A.append(u * p3 - p1)
        A.append(v * p3 - p2)

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    points3D.append(X[:3])

points3D = np.array(points3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], s=5)

scale = 1

Rs = [R1, R2, R3, R4]
ts = [t1, t2, t3, t4]

K_inv = np.linalg.inv(K)
cx, cy = K[0, 2], K[1, 2]

for i, (R, t) in enumerate(zip(Rs, ts), start=1):
    C = -R.T @ t
    C = C.flatten()

    dir_cam = K_inv @ np.array([cx, cy, 1.0])
    dir_world = R.T @ dir_cam
    dir_world = dir_world / np.linalg.norm(dir_world) * scale

    ax.scatter(C[0], C[1], C[2], s=50)
    ax.text(C[0], C[1], C[2], f"C{i}", fontsize=10)

    ax.plot(
        [C[0], C[0] + dir_world[0]],
        [C[1], C[1] + dir_world[1]],
        [C[2], C[2] + dir_world[2]]
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()






# print(len(pts3_unique))
# print(len(pts4_unique))
# H, W = image1.shape[:2]
# scale = 0.5
# H2 = int(H * scale)
# W2 = int(W * scale)
#
# image1_show = cv2.resize(image3, (W2, H2))
# image2_show = cv2.resize(image4, (W2, H2))
#
# canvas = np.zeros((H2, W2 * 2, 3), dtype=np.uint8)
# canvas[:, :W2] = image1_show
# canvas[:, W2:W2 * 2] = image2_show
#
# for p1, p2 in zip(pts3_unique, pts4_unique):
#     x1, y1 = p1
#     x2, y2 = p2
#
#     x1 = int(x1 * scale)
#     y1 = int(y1 * scale)
#     x2 = int(x2 * scale) + W2
#     y2 = int(y2 * scale)
#
#     cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1)
#     cv2.circle(canvas, (x2, y2), 2, (0, 255, 255), -1)
#     cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
# cv2.imshow("Unique Correspondences", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
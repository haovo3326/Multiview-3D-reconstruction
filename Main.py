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
image1_path = os.path.join(root, "Sample", "Image 1.jpg")
image2_path = os.path.join(root, "Sample", "Image 2.jpg")
image3_path = os.path.join(root, "Sample", "Image 3.jpg")
image4_path = os.path.join(root, "Sample", "Image 4.jpg")
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)
image4 = cv2.imread(image4_path)

K = Calibration.calibrate()

# Build DSU for tracking correspondences
R12, t12, matches12 = feature_matching(K, image1, image2)
R23, t23, matches23 = feature_matching(K, image2, image3)
R34, t34, matches34 = feature_matching(K, image3, image4)

dsu = DSU.DSU()
for pt1, pt2 in matches12:
    dsu.union((1, pt1), (2, pt2))
for pt1, pt2 in matches23:
    dsu.union((2, pt1), (3, pt2))
for pt1, pt2 in matches34:
    dsu.union((3, pt1), (4, pt2))
tracks = dsu.groups()

# Reconstructing camera matrix
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
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()






# print(len(pts1_unique))
# print(len(pts2_unique))
# H, W = image1.shape[:2]
# scale = 0.5
# H2 = int(H * scale)
# W2 = int(W * scale)
#
# image1_show = cv2.resize(image2, (W2, H2))
# image2_show = cv2.resize(image3, (W2, H2))
#
# canvas = np.zeros((H2, W2 * 2, 3), dtype=np.uint8)
# canvas[:, :W2] = image1_show
# canvas[:, W2:W2 * 2] = image2_show
#
# for p1, p2 in zip(pts1_unique, pts2_unique):
#     x1, y1 = p1
#     x2, y2 = p2
#
#     x1 = int(x1 * scale)
#     y1 = int(y1 * scale)
#     x2 = int(x2 * scale) + W2
#     y2 = int(y2 * scale)
#
#     cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
#     cv2.circle(canvas, (x2, y2), 3, (0, 0, 255), -1)
#     cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
# cv2.imshow("Unique Correspondences", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
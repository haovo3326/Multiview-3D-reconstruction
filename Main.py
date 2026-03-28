import os
import Utility
import cv2
import numpy as np
import DSU

def feature_matching(img1, img2):
    pts1, pts2 = Utility.get_correspondence(img1, img2)
    pts1_unique, pts2_unique = Utility.rounding_and_unique(pts1, pts2)
    return zip(pts1_unique, pts2_unique)

root = os.getcwd()
image1_path = os.path.join(root, "Sample", "Image 1.jpg")
image2_path = os.path.join(root, "Sample", "Image 2.jpg")
image3_path = os.path.join(root, "Sample", "Image 3.jpg")
image4_path = os.path.join(root, "Sample", "Image 4.jpg")
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)
image4 = cv2.imread(image4_path)

matches12 = feature_matching(image1, image2)
matches23 = feature_matching(image2, image3)
matches34 = feature_matching(image3, image4)

dsu = DSU.DSU()
for pt1, pt2 in matches12:
    dsu.union((1, pt1), (2, pt2))
for pt1, pt2 in matches23:
    dsu.union((2, pt1), (3, pt2))
for pt1, pt2 in matches34:
    dsu.union((3, pt1), (4, pt2))
tracks = dsu.groups()





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
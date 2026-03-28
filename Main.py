import os
import Utility
import cv2
import numpy as np

root = os.getcwd()
image1_path = os.path.join(root, "Sample", "Image 3.jpg")
image2_path = os.path.join(root, "Sample", "Image 4.jpg")
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

pts1, pts2 = Utility.get_correspondence(image1, image2)
pts1_unique, pts2_unique = Utility.rounding_and_unique(pts1, pts2)
print(len(pts1_unique))
print(len(pts2_unique))

H, W = image1.shape[:2]
scale = 0.5
H2 = int(H * scale)
W2 = int(W * scale)

image1_show = cv2.resize(image1, (W2, H2))
image2_show = cv2.resize(image2, (W2, H2))

canvas = np.zeros((H2, W2 * 2, 3), dtype=np.uint8)
canvas[:, :W2] = image1_show
canvas[:, W2:W2 * 2] = image2_show

for p1, p2 in zip(pts1_unique, pts2_unique):
    x1, y1 = p1
    x2, y2 = p2

    x1 = int(x1 * scale)
    y1 = int(y1 * scale)
    x2 = int(x2 * scale) + W2
    y2 = int(y2 * scale)

    cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
    cv2.circle(canvas, (x2, y2), 3, (0, 0, 255), -1)
    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow("Unique Correspondences", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
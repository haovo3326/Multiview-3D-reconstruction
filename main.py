import numpy as np

import calibration
from constructor import Constructor
from optimizer import GD_optimizer, LM_optimizer
import utility

K, _ = calibration.calibrate()
builder = Constructor(K)
#
builder.load_img("Samples/Sample 7/Image 1.jpg")
builder.load_img("Samples/Sample 7/Image 2.jpg")
builder.load_img("Samples/Sample 7/Image 3.jpg")
builder.load_img("Samples/Sample 7/Image 4.jpg")
builder.load_img("Samples/Sample 7/Image 5.jpg")
builder.load_img("Samples/Sample 7/Image 6.jpg")
builder.load_img("Samples/Sample 7/Image 7.jpg")
builder.load_img("Samples/Sample 7/Image 8.jpg")
builder.load_img("Samples/Sample 7/Image 9.jpg")
builder.load_img("Samples/Sample 7/Image 10.jpg")
builder.load_img("Samples/Sample 7/Image 11.jpg")
builder.load_img("Samples/Sample 7/Image 12.jpg")
#
# builder.display_essential_correspondences(2, 3)
#
#
#
builder.construct_anchor()
builder.construct_scene()
# # # # #
# # # # # # -------- OPTIMIZATION --------
optimizer = LM_optimizer(builder)
optimizer.optimize()
# # # # # # # -------- VISUALIZE --------
builder.display_point_cloud()

# pts = []
# for X in builder.track_to_point.values():
#     if X is None:
#         continue
#     pts.append(np.asarray(X).reshape(3))
# utility.save_point_cloud_ply(pts, "output.ply")
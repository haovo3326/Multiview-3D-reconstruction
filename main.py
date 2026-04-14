import numpy as np

import calibration
from constructor import Constructor
from optimizer import GDOptimizer
import utility

K = calibration.calibrate()

builder = Constructor(K)

builder.load_img("Samples/Sample 5/Image 1.png")
builder.load_img("Samples/Sample 5/Image 2.png")
builder.load_img("Samples/Sample 5/Image 3.png")
builder.load_img("Samples/Sample 5/Image 4.png")

# builder.display_essential_correspondences(4, 5)



builder.construct_anchor()
builder.construct_scene()
# # #
# # # # -------- OPTIMIZATION --------
optimizer = GDOptimizer(builder)

optimizer.optimize(5e-7, 1000, 50, 2, 20, 0.005)
# # # #
# # # # # -------- VISUALIZE --------
builder.display_point_cloud()

pts = []
for X in builder.track_to_point.values():
    if X is None:
        continue
    pts.append(np.asarray(X).reshape(3))
utility.save_point_cloud_ply(pts, "output.ply")
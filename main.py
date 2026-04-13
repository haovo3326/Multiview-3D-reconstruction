import calibration
from constructor import Constructor
from optimizer import GDOptimizer

K = calibration.calibrate()

builder = Constructor(K)

builder.load_img("Sample 1/Image 1.png")
builder.load_img("Sample 1/Image 2.png")
builder.load_img("Sample 1/Image 3.png")
builder.load_img("Sample 1/Image 4.png")


builder.construct_anchor()
builder.construct_scene()

# -------- OPTIMIZATION --------
optimizer = GDOptimizer(builder)

optimizer.optimize(6e-7, 1000, 50, 2, 20, 0.005)

# -------- VISUALIZE --------
builder.display_point_cloud()
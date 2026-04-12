import calibration
from constructor import Constructor
from optimizer import GDOptimizer

K = calibration.calibrate()

builder = Constructor(K)

builder.load_img("Sample/Image 1.png")
builder.load_img("Sample/Image 2.png")
builder.load_img("Sample/Image 3.png")
builder.load_img("Sample/Image 4.png")

builder.construct_anchor()
builder.construct_scene()

# -------- OPTIMIZATION --------
optimizer = GDOptimizer(builder)

optimizer.optimize(1e-9, 1000)

# -------- VISUALIZE --------
builder.show_point_cloud()
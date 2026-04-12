import calibration
from constructor import Constructor

K = calibration.calibrate()
builder = Constructor(K)
builder.load_img("Sample/Image 1.png")
builder.load_img("Sample/Image 2.png")
builder.load_img("Sample/Image 3.png")
builder.load_img("Sample/Image 4.png")

builder.construct_anchor()
builder.construct_scene()
builder.show_point_cloud()
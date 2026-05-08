import numpy as np

import calibration
import utility
from constructor import Constructor
from optimizer import GD_optimizer, LM_optimizer, ensure_save_dir, load_loss_log, load_tracks, save_tracks

SAVE_DIR = ensure_save_dir("save")
LOSS_FILE = SAVE_DIR / "loss_log.txt"
TRACKS_FILE = SAVE_DIR / "tracks.pkl"
CAMERAS_FILE = SAVE_DIR / "cameras.json"

K, dist = calibration.calibrate()
builder = Constructor(K, dist)
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

builder.construct_anchor()
builder.construct_scene()
builder.colorize_point_cloud()

optimizer = LM_optimizer(builder)
optimizer.optimize(loss_file=LOSS_FILE)

tracks = optimizer._get_tracks(save_observation_points=True)
save_tracks(tracks, TRACKS_FILE)
utility.save_camera_info(builder.camera_matrices, CAMERAS_FILE)

pts = []
colors = []
for root, X in builder.track_to_point.items():
    if X is None:
        continue
    color = builder.point_to_color.get(root)
    if color is None:
        continue
    pts.append(np.asarray(X).reshape(3))
    colors.append(np.asarray(color).reshape(3))
utility.save_point_cloud_ply(pts, "save/output.ply", colors)

builder.display_point_cloud()

import torch
import numpy as np
import cv2
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import calibration
from dsu import dsu


class Constructor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.images = []
        self.features = []
        self.camera_matrices = []

        self.tracker = dsu()
        self.tracks = []
        self.node_to_track_id = {}

    def load_img(self, img_path):
        image = load_image(img_path).to(self.device)
        feature = self.extractor.extract(image)
        self.images.append(image)
        self.features.append(feature)

    def feature_matching(self, features0, features1):
        matches = self.matcher({'image0': features0, 'image1': features1})
        matches = rbd(matches)
        return matches['matches'].cpu().numpy()

    def compute_pose(self, keypoints0, keypoints1, matches01, K):
        samples0 = keypoints0[matches01[:, 0]]
        samples1 = keypoints1[matches01[:, 1]]

        E, inlier_mask = cv2.findEssentialMat(
            samples0,
            samples1,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        inlier_mask = inlier_mask.ravel().astype(bool)
        matches01 = matches01[inlier_mask]
        samples0 = samples0[inlier_mask]
        samples1 = samples1[inlier_mask]

        _, R, t, pose_mask = cv2.recoverPose(E, samples0, samples1, K)
        pose_mask = pose_mask.ravel().astype(bool)
        matches01 = matches01[pose_mask]

        return matches01, R, t

    def build_projection_matrix(self, K, R, t):
        return (K @ np.hstack((R, t))).astype(np.float64)

    def construct_anchor(self, K):
        features0 = self.features[0]
        features1 = self.features[1]

        keypoints0 = features0['keypoints'][0].cpu().numpy().astype(np.float32)
        keypoints1 = features1['keypoints'][0].cpu().numpy().astype(np.float32)

        matches01 = self.feature_matching(features0, features1)
        matches01, R, t = self.compute_pose(keypoints0, keypoints1, matches01, K)

        P0 = self.build_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
        P1 = self.build_projection_matrix(K, R, t)

        self.camera_matrices.append(P0)
        self.camera_matrices.append(P1)

        # build DSU tracks
        for i0, i1 in matches01:
            node0 = (0, int(i0))
            node1 = (1, int(i1))
            self.tracker.union(node0, node1)

        self.build_track_objects()

        # triangulate for 2-view anchor tracks
        for i0, i1 in matches01:
            node0 = (0, int(i0))
            track_id = self.node_to_track_id[node0]

            pt0 = keypoints0[int(i0)].reshape(2, 1)
            pt1 = keypoints1[int(i1)].reshape(2, 1)

            X_h = cv2.triangulatePoints(P0, P1, pt0, pt1)
            X = (X_h[:3] / X_h[3]).reshape(3)

            self.tracks[track_id]["point3d"] = X
            self.tracks[track_id]["is_computed"] = True

        return self.tracks

    def construct_scene(self, K):
        for i in range(2, len(self.features)):
            features_cur = self.features[i]
            features_pre = self.features[i - 1]

            keypoints_cur = features_cur['keypoints'][0].cpu().numpy().astype(np.float32)
            keypoints_pre = features_pre['keypoints'][0].cpu().numpy().astype(np.float32)

            matches_pre_cur, _, _ = self.feature_matching(features_pre, features_cur)


    def build_track_objects(self):
        groups = self.tracker.groups()
        self.tracks = []
        self.node_to_track_id = {}

        for track_id, group in enumerate(groups):
            track = {
                "track_id": track_id,
                "observations": sorted(group, key=lambda x: x[0]),
                "point3d": None,        # luôn có slot
                "is_computed": False    # mặc định chưa tính
            }
            self.tracks.append(track)

            for node in group:
                self.node_to_track_id[node] = track_id

    def print_tracks(self, max_tracks=20):
        for track in self.tracks[:max_tracks]:
            print(f"Track ID: {track['track_id']}")
            print(f"Observations: {track['observations']}")
            print(f"Point3D: {track['point3d']}")
            print(f"Computed: {track['is_computed']}")
            print("-" * 40)


# Testing
K = calibration.calibrate()
builder = Constructor()
builder.load_img("Sample/Image 1.png")
builder.load_img("Sample/Image 2.png")

tracks = builder.construct_anchor(K)
builder.print_tracks()
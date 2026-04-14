import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utility

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from dsu import dsu


class Constructor:
    def __init__(self, K):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.images = []
        self.features = []
        self.K = K.astype(np.float64)
        self.camera_matrices = []   # each item: {"q": [w,x,y,z], "t": (3,)}

        self.tracker = dsu()
        self.track_to_point = {}

    def load_img(self, img_path):
        image = load_image(img_path).to(self.device)

        # image = utility.normalize_brightness(image)  # <-- ADD THIS

        image = image.to(self.device)
        feature = self.extractor.extract(image)

        self.features.append(feature)
        self.images.append(image)

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

        return matches01, R.astype(np.float64), t.astype(np.float64)

    def remap_track_points(self):
        new_map = {}
        for old_root, X in self.track_to_point.items():
            new_root = self.tracker.find(old_root)
            if new_root not in new_map:
                new_map[new_root] = X
        return new_map

    def construct_anchor(self):
        features0 = self.features[0]
        features1 = self.features[1]

        matches01 = self.feature_matching(features0, features1)
        keypoints0 = features0['keypoints'][0].cpu().numpy().astype(np.float32)
        keypoints1 = features1['keypoints'][0].cpu().numpy().astype(np.float32)

        matches01, R, t = self.compute_pose(keypoints0, keypoints1, matches01, self.K)
        quaternion = utility.R_to_quaternion(R)

        for i0, i1 in matches01:
            self.tracker.union((0, int(i0)), (1, int(i1)))

        P0 = utility.build_projection_matrix(self.K, np.eye(3), np.zeros((3, 1)))
        P1 = utility.build_projection_matrix(self.K, R, t)

        pts0 = keypoints0[matches01[:, 0]]
        pts1 = keypoints1[matches01[:, 1]]
        X01 = utility.triangulate_points(P0, P1, pts0, pts1)

        self.camera_matrices.append({
            "q": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "t": np.zeros(3, dtype=np.float64)
        })
        self.camera_matrices.append({
            "q": quaternion.astype(np.float64),
            "t": t.reshape(3).astype(np.float64)
        })

        for j, (i0, i1) in enumerate(matches01):
            root = self.tracker.find((0, int(i0)))
            self.track_to_point[root] = np.asarray(X01[j], dtype=np.float64).reshape(3)

    def construct_scene(self):
        for i in range(2, len(self.features)):
            features_pre = self.features[i - 1]
            features_cur = self.features[i]

            matches_pre_cur = self.feature_matching(features_pre, features_cur)
            keypoints_pre = features_pre['keypoints'][0].cpu().numpy().astype(np.float32)
            keypoints_cur = features_cur['keypoints'][0].cpu().numpy().astype(np.float32)

            matches_pre_cur, _, _ = self.compute_pose(
                keypoints_pre, keypoints_cur, matches_pre_cur, self.K
            )

            obj_points = []
            img_points = []

            for i_pre, i_cur in matches_pre_cur:
                node_pre = (i - 1, int(i_pre))
                if node_pre not in self.tracker.parent:
                    continue

                root = self.tracker.find(node_pre)
                if root not in self.track_to_point:
                    continue

                obj_points.append(self.track_to_point[root])
                img_points.append(keypoints_cur[int(i_cur)])

            obj_points = np.asarray(obj_points, dtype=np.float32)
            img_points = np.asarray(img_points, dtype=np.float32)

            if len(obj_points) < 6:
                raise ValueError(f"Not enough 2D-3D correspondences for PnP on image at index {i}")

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=obj_points,
                imagePoints=img_points,
                cameraMatrix=self.K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=8.0,
                confidence=0.999,
                iterationsCount=1000
            )

            if not ok:
                raise ValueError(f"PnP failed for image at index {i}")

            R_cur, _ = cv2.Rodrigues(rvec)
            t_cur = tvec.reshape(3, 1).astype(np.float64)
            q_cur = utility.R_to_quaternion(R_cur)

            self.camera_matrices.append({
                "q": q_cur.astype(np.float64),
                "t": t_cur.reshape(3).astype(np.float64)
            })

            R_pre = utility.quaternion_to_R(self.camera_matrices[i - 1]["q"])
            t_pre = self.camera_matrices[i - 1]["t"].reshape(3, 1)

            P_pre = utility.build_projection_matrix(self.K, R_pre, t_pre)
            P_cur = utility.build_projection_matrix(self.K, R_cur, t_cur)

            for i_pre, i_cur in matches_pre_cur:
                self.tracker.union((i - 1, int(i_pre)), (i, int(i_cur)))

            self.track_to_point = self.remap_track_points()

            pts_pre = keypoints_pre[matches_pre_cur[:, 0]]
            pts_cur = keypoints_cur[matches_pre_cur[:, 1]]
            X_pre_cur = utility.triangulate_points(P_pre, P_cur, pts_pre, pts_cur)

            for j, (i_pre, i_cur) in enumerate(matches_pre_cur):
                root = self.tracker.find((i - 1, int(i_pre)))
                if root not in self.track_to_point:
                    self.track_to_point[root] = np.asarray(X_pre_cur[j], dtype=np.float64).reshape(3)

    def display_point_cloud(self):
        pts = []

        for X in self.track_to_point.values():
            if X is None:
                continue

            X = np.asarray(X, dtype=np.float64).reshape(-1)
            if X.shape[0] != 3:
                continue
            if not np.all(np.isfinite(X)):
                continue

            pts.append(X)

        if len(pts) == 0:
            print("No valid 3D points to display.")
            return

        pts = np.asarray(pts, dtype=np.float64)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Point Cloud")

        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        z_mid = (z_min + z_max) / 2.0

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        if max_range == 0:
            max_range = 1.0

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

        plt.show()

    def display_essential_correspondences(self, idx0, idx1):
        image0 = self.images[idx0]
        image1 = self.images[idx1]

        # to numpy (H,W,3)
        img0 = image0.permute(1, 2, 0).cpu().numpy()
        img1 = image1.permute(1, 2, 0).cpu().numpy()

        features0 = self.features[idx0]
        features1 = self.features[idx1]

        keypoints0 = features0["keypoints"][0].cpu().numpy()
        keypoints1 = features1["keypoints"][0].cpu().numpy()

        matches01 = self.feature_matching(features0, features1)

        pts0 = keypoints0[matches01[:, 0]]
        pts1 = keypoints1[matches01[:, 1]]

        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            print("E failed")
            return

        inlier_mask = inlier_mask.ravel().astype(bool)

        pts0_in = pts0[inlier_mask]
        pts1_in = pts1[inlier_mask]

        print(f"Total matches: {len(matches01)}")
        print(f"Inliers (E satisfied): {len(pts0_in)}")

        # stack images
        h = max(img0.shape[0], img1.shape[0])
        w0 = img0.shape[1]
        w1 = img1.shape[1]

        canvas = np.zeros((h, w0 + w1, 3), dtype=img0.dtype)
        canvas[:img0.shape[0], :w0] = img0
        canvas[:img1.shape[0], w0:w0 + w1] = img1

        # draw ALL inlier correspondences
        for (p0, p1) in zip(pts0_in, pts1_in):
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]) + w0, int(p1[1])

            cv2.circle(canvas, (x0, y0), 2, (0, 255, 0), -1)
            cv2.circle(canvas, (x1, y1), 2, (0, 255, 0), -1)
            cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)

        plt.figure(figsize=(12, 6))
        plt.imshow(canvas)
        plt.axis("off")
        plt.show()
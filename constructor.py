import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt

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
        if len(matches01) == 0:
            return np.empty((0, 2), dtype=int), None, None

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

        if E is None or inlier_mask is None:
            return np.empty((0, 2), dtype=int), None, None

        inlier_mask = inlier_mask.ravel().astype(bool)
        matches01 = matches01[inlier_mask]
        samples0 = samples0[inlier_mask]
        samples1 = samples1[inlier_mask]

        if len(matches01) == 0:
            return np.empty((0, 2), dtype=int), None, None

        _, R, t, pose_mask = cv2.recoverPose(E, samples0, samples1, K)
        pose_mask = pose_mask.ravel().astype(bool)
        matches01 = matches01[pose_mask]

        if len(matches01) == 0:
            return np.empty((0, 2), dtype=int), None, None

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

            if self.tracks[track_id]["is_computed"]:
                continue

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

            # raw matches
            matches_pre_cur = self.feature_matching(features_pre, features_cur)

            # keep only geometric inliers between pre and cur
            matches_pre_cur, _, _ = self.compute_pose(
                keypoints_pre,
                keypoints_cur,
                matches_pre_cur,
                K
            )

            if len(matches_pre_cur) == 0:
                print(f"Image {i}: no valid pre-cur matches")
                continue

            # split into:
            # 1) old-track matches -> for PnP
            # 2) new matches -> for triangulation
            object_points = []
            image_points = []
            tracked_pairs = []  # (track_id, idx_cur)
            new_matches = []  # (idx_pre, idx_cur)

            for idx_pre, idx_cur in matches_pre_cur:
                node_pre = (i - 1, int(idx_pre))

                if node_pre in self.node_to_track_id:
                    track_id = self.node_to_track_id[node_pre]
                    track = self.tracks[track_id]

                    object_points.append(track["point3d"])
                    image_points.append(keypoints_cur[int(idx_cur)])
                    tracked_pairs.append((track_id, int(idx_cur)))
                else:
                    new_matches.append((int(idx_pre), int(idx_cur)))

            object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
            image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)

            # need pose to triangulate new matches
            if len(object_points) < 4:
                print(f"Image {i}: not enough old-track correspondences for PnP")
                continue

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                None,
                reprojectionError=8.0,
                confidence=0.999,
                iterationsCount=1000,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success or inliers is None or len(inliers) < 4:
                print(f"Image {i}: PnP failed")
                continue

            inlier_idx = inliers.ravel()
            tracked_pairs_inlier = [tracked_pairs[j] for j in inlier_idx]

            R, _ = cv2.Rodrigues(rvec)
            P_cur = self.build_projection_matrix(K, R, tvec)
            P_pre = self.camera_matrices[i - 1]
            self.camera_matrices.append(P_cur)

            # save old 3D points before rebuilding tracks
            old_node_to_point3d = {}
            for track in self.tracks:
                if track["is_computed"] and track["point3d"] is not None:
                    for node in track["observations"]:
                        old_node_to_point3d[node] = track["point3d"]

            # merge PnP inlier current observations into old tracks
            for track_id, idx_cur in tracked_pairs_inlier:
                node_cur = (i, int(idx_cur))
                old_node = self.tracks[track_id]["observations"][0]
                self.tracker.union(old_node, node_cur)

            # create new tracks from unseen matches
            for idx_pre, idx_cur in new_matches:
                node_pre = (i - 1, int(idx_pre))
                node_cur = (i, int(idx_cur))
                self.tracker.union(node_pre, node_cur)

            # rebuild
            self.build_track_objects()

            # restore old computed points
            for track in self.tracks:
                for node in track["observations"]:
                    if node in old_node_to_point3d:
                        track["point3d"] = old_node_to_point3d[node]
                        track["is_computed"] = True
                        break

            # triangulate only the truly new tracks
            triangulated_track_ids = set()

            for idx_pre, idx_cur in new_matches:
                node_pre = (i - 1, int(idx_pre))
                track_id = self.node_to_track_id[node_pre]
                track = self.tracks[track_id]

                if track["is_computed"] or track_id in triangulated_track_ids:
                    continue

                pt_pre = keypoints_pre[int(idx_pre)].reshape(2, 1)
                pt_cur = keypoints_cur[int(idx_cur)].reshape(2, 1)

                X_h = cv2.triangulatePoints(P_pre, P_cur, pt_pre, pt_cur)
                X = (X_h[:3] / X_h[3]).reshape(3)

                track["point3d"] = X
                track["is_computed"] = True
                triangulated_track_ids.add(track_id)

            print(f"Image {i + 1}: PnP inliers = {len(inlier_idx)}")
            print(f"Image {i + 1}: new matches triangulated = {len(triangulated_track_ids)}")


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

    def show_point_cloud(self):
        points = []

        for track in self.tracks:
            if track["is_computed"] and track["point3d"] is not None:
                points.append(track["point3d"])

        if len(points) == 0:
            print("No 3D points to display")
            return

        points = np.array(points)  # (N, 3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


# Testing
K = calibration.calibrate()
builder = Constructor()
builder.load_img("Sample/Image 1.png")
builder.load_img("Sample/Image 2.png")
builder.load_img("Sample/Image 3.png")
builder.load_img("Sample/Image 4.png")
builder.construct_anchor(K)
builder.construct_scene(K)
builder.show_point_cloud()
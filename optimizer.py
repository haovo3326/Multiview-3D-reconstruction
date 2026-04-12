import numpy as np


class GDOptimizer:
    def __init__(self, constructor):
        self.constructor = constructor

    def _normalize_quaternion(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too close to zero.")
        return q / n, n

    def _quat_norm_jacobian(self, q_raw):
        """
        q_unit = q_raw / ||q_raw||
        Return J = dq_unit / dq_raw, shape (4, 4)
        """
        q_raw = np.asarray(q_raw, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q_raw)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too close to zero.")

        q_unit = q_raw / n
        I = np.eye(4, dtype=np.float64)
        J = (I - np.outer(q_unit, q_unit)) / n
        return J

    def backprop(self):
        tracks = self.constructor.tracks

        global_X_grads = {}
        global_t_grads = {}
        global_q_grads = {}
        global_cam_count = {}  # NEW

        K = self.constructor.K.astype(np.float64)

        for track_id, track in enumerate(tracks):
            X = track["point3d"]
            observations = track["observations"]

            X_vec = np.asarray(X, dtype=np.float64).reshape(3)
            weight = 1.0 / len(observations)

            if track_id not in global_X_grads:
                global_X_grads[track_id] = np.zeros(3, dtype=np.float64)

            for (img_id, kp_id) in observations:
                features = self.constructor.features[img_id]
                keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                x_ij = keypoints[int(kp_id)].reshape(2)

                cam = self.constructor.camera_matrices[img_id]

                q_raw = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                q_unit, _ = self._normalize_quaternion(q_raw)

                x, y, z, w = q_unit

                R = self.constructor.quaternion_to_rotation_matrix(q_unit).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                Y = K @ Z

                x_hat = np.array([Y[0] / Y[2], Y[1] / Y[2]], dtype=np.float64)
                r_ij = x_ij - x_hat

                dL_dr = 2.0 * weight * r_ij

                dr_dY = np.array([
                    [-1.0 / Y[2], 0.0, Y[0] / (Y[2] ** 2)],
                    [0.0, -1.0 / Y[2], Y[1] / (Y[2] ** 2)]
                ], dtype=np.float64)

                dL_dY = dr_dY.T @ dL_dr
                dL_dZ = K.T @ dL_dY

                dL_dX = R.T @ dL_dZ
                dL_dt = dL_dZ.copy()
                dL_dR = np.outer(dL_dZ, X_vec)

                # quaternion grads (same as before)
                dR_dx_prime = np.array([
                    [0.0, 2.0 * y, 2.0 * z],
                    [2.0 * y, -4.0 * x, -2.0 * w],
                    [2.0 * z, 2.0 * w, -4.0 * x]
                ])

                dR_dy_prime = np.array([
                    [-4.0 * y, 2.0 * x, 2.0 * w],
                    [2.0 * x, 0.0, 2.0 * z],
                    [-2.0 * w, 2.0 * z, -4.0 * y]
                ])

                dR_dz_prime = np.array([
                    [-4.0 * z, -2.0 * w, 2.0 * x],
                    [2.0 * w, -4.0 * z, 2.0 * y],
                    [2.0 * x, 2.0 * y, 0.0]
                ])

                dR_dw_prime = np.array([
                    [0.0, -2.0 * z, 2.0 * y],
                    [2.0 * z, 0.0, -2.0 * x],
                    [-2.0 * y, 2.0 * x, 0.0]
                ])

                dL_dq_prime = np.array([
                    np.sum(dL_dR * dR_dx_prime),
                    np.sum(dL_dR * dR_dy_prime),
                    np.sum(dL_dR * dR_dz_prime),
                    np.sum(dL_dR * dR_dw_prime)
                ])

                J_norm = self._quat_norm_jacobian(q_raw)
                dL_dq_raw = J_norm.T @ dL_dq_prime

                # accumulate
                global_X_grads[track_id] += dL_dX

                if img_id not in global_t_grads:
                    global_t_grads[img_id] = np.zeros(3)
                    global_q_grads[img_id] = np.zeros(4)
                    global_cam_count[img_id] = 0  # NEW

                global_t_grads[img_id] += dL_dt
                global_q_grads[img_id] += dL_dq_raw
                global_cam_count[img_id] += 1  # NEW

        # -------- NORMALIZATION --------
        for img_id in global_cam_count:
            count = global_cam_count[img_id]
            if count > 0:
                global_t_grads[img_id] /= count
                global_q_grads[img_id] /= count

        return global_X_grads, global_t_grads, global_q_grads

    def loss(self):
        tracks = self.constructor.tracks
        K = self.constructor.K.astype(np.float64)
        total_loss = 0.0
        valid_track_count = 0

        for track in tracks:
            X = track["point3d"]
            observations = track["observations"]

            if X is None or observations is None or len(observations) == 0:
                continue

            X_vec = np.asarray(X, dtype=np.float64).reshape(3)
            weight = 1.0 / len(observations)

            track_loss = 0.0

            for (img_id, kp_id) in observations:
                features = self.constructor.features[img_id]
                keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                x_ij = keypoints[int(kp_id)].reshape(2)

                cam = self.constructor.camera_matrices[img_id]

                q_raw = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                q_unit, _ = self._normalize_quaternion(q_raw)

                R = self.constructor.quaternion_to_rotation_matrix(q_unit).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                Y = K @ Z

                x_hat = np.array([
                    Y[0] / Y[2],
                    Y[1] / Y[2]
                ], dtype=np.float64)

                r_ij = x_ij - x_hat
                track_loss += np.sqrt(weight * np.sum(r_ij ** 2))

            total_loss += track_loss
            valid_track_count += 1

        if valid_track_count == 0:
            return 0.0

        total_loss /= valid_track_count
        return total_loss

    def optimize(self, lr=1e-4, iters=100, patience=10):
        best_loss = self.loss()
        best_tracks = []
        best_cameras = []

        # save initial best state
        for track in self.constructor.tracks:
            track_copy = dict(track)
            if track["point3d"] is None:
                track_copy["point3d"] = None
            else:
                track_copy["point3d"] = np.asarray(track["point3d"], dtype=np.float64).copy()
            best_tracks.append(track_copy)

        for cam in self.constructor.camera_matrices:
            best_cameras.append({
                "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                "t": np.asarray(cam["t"], dtype=np.float64).copy()
            })

        wait = 0

        for step in range(iters):
            print(f"Iteration {step + 1}/{iters}...")

            global_X_grads, global_t_grads, global_q_grads = self.backprop()

            # backup current state before trying update
            old_tracks = []
            old_cameras = []

            for track in self.constructor.tracks:
                track_copy = dict(track)
                if track["point3d"] is None:
                    track_copy["point3d"] = None
                else:
                    track_copy["point3d"] = np.asarray(track["point3d"], dtype=np.float64).copy()
                old_tracks.append(track_copy)

            for cam in self.constructor.camera_matrices:
                old_cameras.append({
                    "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                    "t": np.asarray(cam["t"], dtype=np.float64).copy()
                })

            # update 3D points
            for track_id, grad_X in global_X_grads.items():
                X = self.constructor.tracks[track_id]["point3d"]
                if X is None:
                    continue

                X = np.asarray(X, dtype=np.float64).reshape(3)
                X = X - lr * grad_X
                self.constructor.tracks[track_id]["point3d"] = X

            # update camera parameters
            for img_id, cam in enumerate(self.constructor.camera_matrices):
                if img_id in global_t_grads:
                    t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
                    t = t - lr * global_t_grads[img_id]
                    cam["t"] = t

                if img_id in global_q_grads:
                    q = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                    q = q - lr * global_q_grads[img_id]
                    q, _ = self._normalize_quaternion(q)
                    cam["q"] = q

            current_loss = self.loss()
            print("Reprojection loss:", current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                wait = 0

                best_tracks = []
                best_cameras = []

                for track in self.constructor.tracks:
                    track_copy = dict(track)
                    if track["point3d"] is None:
                        track_copy["point3d"] = None
                    else:
                        track_copy["point3d"] = np.asarray(track["point3d"], dtype=np.float64).copy()
                    best_tracks.append(track_copy)

                for cam in self.constructor.camera_matrices:
                    best_cameras.append({
                        "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                        "t": np.asarray(cam["t"], dtype=np.float64).copy()
                    })
            else:
                wait += 1

                # revert parameters
                for i, track in enumerate(old_tracks):
                    self.constructor.tracks[i]["point3d"] = track["point3d"]

                for i, cam in enumerate(old_cameras):
                    self.constructor.camera_matrices[i]["q"] = cam["q"]
                    self.constructor.camera_matrices[i]["t"] = cam["t"]

                if wait >= patience:
                    print(f"Early stopping at iteration {step + 1}")
                    break

        # restore best state at the end
        for i, track in enumerate(best_tracks):
            self.constructor.tracks[i]["point3d"] = track["point3d"]

        for i, cam in enumerate(best_cameras):
            self.constructor.camera_matrices[i]["q"] = cam["q"]
            self.constructor.camera_matrices[i]["t"] = cam["t"]

        print("Best reprojection loss:", best_loss)

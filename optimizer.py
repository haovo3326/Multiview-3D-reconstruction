import numpy as np
from collections import deque
import utility


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
        q_raw = np.asarray(q_raw, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q_raw)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too close to zero.")

        q_unit = q_raw / n
        I = np.eye(4, dtype=np.float64)
        J = (I - np.outer(q_unit, q_unit)) / n
        return J

    def _get_tracks(self):
        groups = self.constructor.tracker.groups()
        tracks = []

        for group in groups:
            if len(group) == 0:
                continue

            root = self.constructor.tracker.find(group[0])
            X = self.constructor.track_to_point.get(root)

            if X is None:
                continue

            tracks.append({
                "root": root,
                "point3d": np.asarray(X, dtype=np.float64).reshape(3),
                "observations": group
            })

        return tracks

    def backprop(self):
        tracks = self._get_tracks()

        global_X_grads = {}
        global_t_grads = {}
        global_q_grads = {}
        global_cam_count = {}

        K = self.constructor.K.astype(np.float64)

        for track in tracks:
            root = track["root"]
            X_vec = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
            observations = track["observations"]

            if observations is None or len(observations) == 0:
                continue

            weight = 1.0 / len(observations)

            if root not in global_X_grads:
                global_X_grads[root] = np.zeros(3, dtype=np.float64)

            for (img_id, kp_id) in observations:
                features = self.constructor.features[img_id]
                keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                x_ij = keypoints[int(kp_id)].reshape(2)

                cam = self.constructor.camera_matrices[img_id]

                q_raw = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                q_unit, _ = self._normalize_quaternion(q_raw)

                w, x, y, z = q_unit
                R = utility.quaternion_to_R(q_unit).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                Y = K @ Z

                if abs(Y[2]) < 1e-12:
                    continue

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

                dR_dw_prime = np.array([
                    [0.0, -2.0 * z, 2.0 * y],
                    [2.0 * z, 0.0, -2.0 * x],
                    [-2.0 * y, 2.0 * x, 0.0]
                ], dtype=np.float64)

                dR_dx_prime = np.array([
                    [0.0, 2.0 * y, 2.0 * z],
                    [2.0 * y, -4.0 * x, -2.0 * w],
                    [2.0 * z, 2.0 * w, -4.0 * x]
                ], dtype=np.float64)

                dR_dy_prime = np.array([
                    [-4.0 * y, 2.0 * x, 2.0 * w],
                    [2.0 * x, 0.0, 2.0 * z],
                    [-2.0 * w, 2.0 * z, -4.0 * y]
                ], dtype=np.float64)

                dR_dz_prime = np.array([
                    [-4.0 * z, -2.0 * w, 2.0 * x],
                    [2.0 * w, -4.0 * z, 2.0 * y],
                    [2.0 * x, 2.0 * y, 0.0]
                ], dtype=np.float64)

                dL_dq_prime = np.array([
                    np.sum(dL_dR * dR_dw_prime),
                    np.sum(dL_dR * dR_dx_prime),
                    np.sum(dL_dR * dR_dy_prime),
                    np.sum(dL_dR * dR_dz_prime)
                ], dtype=np.float64)

                J_norm = self._quat_norm_jacobian(q_raw)
                dL_dq_raw = J_norm.T @ dL_dq_prime

                global_X_grads[root] += dL_dX

                if img_id not in global_t_grads:
                    global_t_grads[img_id] = np.zeros(3, dtype=np.float64)
                    global_q_grads[img_id] = np.zeros(4, dtype=np.float64)
                    global_cam_count[img_id] = 0

                global_t_grads[img_id] += dL_dt
                global_q_grads[img_id] += dL_dq_raw
                global_cam_count[img_id] += 1

        for img_id in global_cam_count:
            count = global_cam_count[img_id]
            if count > 0:
                global_t_grads[img_id] /= count
                global_q_grads[img_id] /= count

        return global_X_grads, global_t_grads, global_q_grads

    def loss(self):
        tracks = self._get_tracks()
        K = self.constructor.K.astype(np.float64)

        total_loss = 0.0
        valid_track_count = 0

        for track in tracks:
            X_vec = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
            observations = track["observations"]

            if observations is None or len(observations) == 0:
                continue

            weight = 1.0 / len(observations)
            track_loss = 0.0

            for (img_id, kp_id) in observations:
                features = self.constructor.features[img_id]
                keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                x_ij = keypoints[int(kp_id)].reshape(2)

                cam = self.constructor.camera_matrices[img_id]

                q_raw = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                q_unit, _ = self._normalize_quaternion(q_raw)

                R = utility.quaternion_to_R(q_unit).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                Y = K @ Z

                if abs(Y[2]) < 1e-12:
                    continue

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

        return total_loss / valid_track_count

    def optimize(self, lr=1e-4, iters=100, patience=10,
                 scale=0.5, history=30, threshold=0.01,
                 eps=1e-12, loss_file="loss_log.txt"):
        best_loss = self.loss()

        best_track_to_point = {
            root: np.asarray(X, dtype=np.float64).copy()
            for root, X in self.constructor.track_to_point.items()
        }

        best_cameras = []
        for cam in self.constructor.camera_matrices:
            best_cameras.append({
                "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                "t": np.asarray(cam["t"], dtype=np.float64).copy()
            })

        loss_track = deque(maxlen=history)
        loss_track.append(best_loss)

        wait = 0

        with open(loss_file, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
            f.write(f"-1,{best_loss}\n")
            f.flush()

            for step in range(iters):
                print(f"Iteration {step + 1}/{iters}...")

                global_X_grads, global_t_grads, global_q_grads = self.backprop()

                old_track_to_point = {
                    root: np.asarray(X, dtype=np.float64).copy()
                    for root, X in self.constructor.track_to_point.items()
                }

                old_cameras = []
                for cam in self.constructor.camera_matrices:
                    old_cameras.append({
                        "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                        "t": np.asarray(cam["t"], dtype=np.float64).copy()
                    })

                for root, grad_X in global_X_grads.items():
                    current_root = self.constructor.tracker.find(root)
                    if current_root not in self.constructor.track_to_point:
                        continue

                    X = np.asarray(self.constructor.track_to_point[current_root], dtype=np.float64).reshape(3)
                    X = X - lr * grad_X
                    self.constructor.track_to_point[current_root] = X

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

                f.write(f"{step},{current_loss}\n")
                f.flush()

                loss_track.append(current_loss)

                if len(loss_track) >= 2:
                    losses = np.array(loss_track, dtype=np.float64)
                    prev_losses = losses[:-1]
                    next_losses = losses[1:]
                    rel_improvements = (prev_losses - next_losses) / np.maximum(np.abs(prev_losses), eps)

                    mean_improve = np.mean(rel_improvements)

                    if len(loss_track) == history and mean_improve < threshold:
                        new_lr = lr * scale
                        if new_lr < lr:
                            print(f"Improvement is small. Scaling lr: {lr:.6e} -> {new_lr:.6e}")
                            lr = new_lr
                        loss_track.clear()
                        loss_track.append(current_loss)

                if current_loss < best_loss:
                    best_loss = current_loss
                    wait = 0

                    best_track_to_point = {
                        root: np.asarray(X, dtype=np.float64).copy()
                        for root, X in self.constructor.track_to_point.items()
                    }

                    best_cameras = []
                    for cam in self.constructor.camera_matrices:
                        best_cameras.append({
                            "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                            "t": np.asarray(cam["t"], dtype=np.float64).copy()
                        })
                else:
                    wait += 1
                    print(f"Waiting {wait}/{patience}")

                    # self.constructor.track_to_point = {
                    #     root: np.asarray(X, dtype=np.float64).copy()
                    #     for root, X in old_track_to_point.items()
                    # }
                    #
                    # for i, cam in enumerate(old_cameras):
                    #     self.constructor.camera_matrices[i]["q"] = cam["q"]
                    #     self.constructor.camera_matrices[i]["t"] = cam["t"]

                    if wait >= patience:
                        print(f"Early stopping at iteration {step + 1}")
                        break

        self.constructor.track_to_point = {
            root: np.asarray(X, dtype=np.float64).copy()
            for root, X in best_track_to_point.items()
        }

        for i, cam in enumerate(best_cameras):
            self.constructor.camera_matrices[i]["q"] = cam["q"]
            self.constructor.camera_matrices[i]["t"] = cam["t"]

        print("Best reprojection loss:", best_loss)
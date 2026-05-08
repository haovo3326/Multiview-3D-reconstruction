import numpy as np
from collections import deque
from pathlib import Path
import pickle
import utility


def ensure_save_dir(save_dir="save"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def save_tracks(tracks, path="save/tracks.pkl"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tracks(path="save/tracks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_loss_log(path="save/loss_log.txt"):
    return np.genfromtxt(path, delimiter=",", names=True)


class GD_optimizer:
    def __init__(self, constructor):
        self.constructor = constructor

    def _project_with_radial(self, k, d, Z):
        fx, fy, cx, cy, s = np.asarray(k, dtype=np.float64).reshape(5)
        k1, k2, k3 = np.asarray(d, dtype=np.float64).reshape(3)
        Z0, Z1, Z2 = np.asarray(Z, dtype=np.float64).reshape(3)

        if abs(Z2) < 1e-12:
            return None

        xn = Z0 / Z2
        yn = Z1 / Z2
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
        xd = xn * radial
        yd = yn * radial

        x_hat = np.array([
            fx * xd + s * yd + cx,
            fy * yd + cy
        ], dtype=np.float64)

        return x_hat, xn, yn, r2, radial, xd, yd

    def _quat_norm_jacobian(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too close to zero.")

        q_prime = q / n
        I = np.eye(4, dtype=np.float64)
        J = (I - np.outer(q_prime, q_prime)) / n
        return J

    def _get_tracks(self, save_observation_points=False):
        groups = self.constructor.tracker.groups()
        tracks = []

        for group in groups:
            if len(group) == 0:
                continue

            root = self.constructor.tracker.find(group[0])
            X = self.constructor.track_to_point.get(root)

            if X is None:
                continue

            if save_observation_points:
                observations = []
                for img_id, kp_id in group:
                    features = self.constructor.features[img_id]
                    keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                    observations.append({
                        "camera_id": int(img_id),
                        "point2d": keypoints[int(kp_id)].reshape(2)
                    })
            else:
                observations = group

            tracks.append({
                "root": root,
                "point3d": np.asarray(X, dtype=np.float64).reshape(3),
                "observations": observations
            })

        return tracks

    def backprop(self):
        tracks = self._get_tracks()

        global_X_grads = {}
        global_t_grads = {}
        global_q_grads = {}
        global_d_grads = {}
        global_cam_count = {}

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

                k = np.asarray(cam.get("k", self.constructor.intrinsic_parameters), dtype=np.float64).reshape(5)
                d = np.asarray(cam.get("d", self.constructor.radial_parameters), dtype=np.float64).reshape(3)
                q = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                q_norm = np.linalg.norm(q)
                if q_norm < 1e-12:
                    continue

                q_prime = q / q_norm
                w, x, y, z = q_prime

                R = utility.quaternion_to_R(q).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                projected = self._project_with_radial(k, d, Z)
                if projected is None:
                    continue

                x_hat, xn, yn, r2, radial, _, _ = projected
                r_ij = x_ij - x_hat

                fx, fy, _, _, s = k
                k1, k2, k3 = d
                Z0, Z1, Z2 = Z

                radial_slope = k1 + 2.0 * k2 * r2 + 3.0 * k3 * (r2 ** 2)
                dradial_dxn = 2.0 * xn * radial_slope
                dradial_dyn = 2.0 * yn * radial_slope

                dxd_dxn = radial + xn * dradial_dxn
                dxd_dyn = xn * dradial_dyn
                dyd_dxn = yn * dradial_dxn
                dyd_dyn = radial + yn * dradial_dyn

                dproj_dnorm = np.array([
                    [fx * dxd_dxn + s * dyd_dxn, fx * dxd_dyn + s * dyd_dyn],
                    [fy * dyd_dxn, fy * dyd_dyn]
                ], dtype=np.float64)

                dnorm_dZ = np.array([
                    [1.0 / Z2, 0.0, -Z0 / (Z2 ** 2)],
                    [0.0, 1.0 / Z2, -Z1 / (Z2 ** 2)]
                ], dtype=np.float64)

                dproj_dZ = dproj_dnorm @ dnorm_dZ
                dL_dxhat = -2.0 * weight * r_ij
                dL_dZ = dproj_dZ.T @ dL_dxhat

                radial_powers = np.array([r2, r2 ** 2, r2 ** 3], dtype=np.float64)
                dL_dd = np.zeros(3, dtype=np.float64)
                for j, power in enumerate(radial_powers):
                    dxd_dd = xn * power
                    dyd_dd = yn * power
                    dproj_dd = np.array([
                        fx * dxd_dd + s * dyd_dd,
                        fy * dyd_dd
                    ], dtype=np.float64)
                    dL_dd[j] = dproj_dd @ dL_dxhat

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

                J_norm = self._quat_norm_jacobian(q)
                dL_dq = J_norm.T @ dL_dq_prime

                global_X_grads[root] += dL_dX

                if img_id not in global_t_grads:
                    global_t_grads[img_id] = np.zeros(3, dtype=np.float64)
                    global_q_grads[img_id] = np.zeros(4, dtype=np.float64)
                    global_d_grads[img_id] = np.zeros(3, dtype=np.float64)
                    global_cam_count[img_id] = 0

                global_t_grads[img_id] += dL_dt
                global_q_grads[img_id] += dL_dq
                global_d_grads[img_id] += dL_dd
                global_cam_count[img_id] += 1

        for img_id in global_cam_count:
            count = global_cam_count[img_id]
            if count > 0:
                global_t_grads[img_id] /= count
                global_q_grads[img_id] /= count
                global_d_grads[img_id] /= count

        return global_X_grads, global_t_grads, global_q_grads, global_d_grads

    def loss(self):
        tracks = self._get_tracks()

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

                k = np.asarray(cam.get("k", self.constructor.intrinsic_parameters), dtype=np.float64).reshape(5)
                d = np.asarray(cam.get("d", self.constructor.radial_parameters), dtype=np.float64).reshape(3)
                q = np.asarray(cam["q"], dtype=np.float64).reshape(4)

                R = utility.quaternion_to_R(q).astype(np.float64)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                Z = R @ X_vec + t
                projected = self._project_with_radial(k, d, Z)
                if projected is None:
                    continue

                x_hat = projected[0]

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
        loss_path = Path(loss_file)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        best_loss = self.loss()

        best_track_to_point = {
            root: np.asarray(X, dtype=np.float64).copy()
            for root, X in self.constructor.track_to_point.items()
        }

        best_cameras = []
        for cam in self.constructor.camera_matrices:
            best_cameras.append({
                "k": np.asarray(cam["k"], dtype=np.float64).copy(),
                "d": np.asarray(cam["d"], dtype=np.float64).copy(),
                "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                "t": np.asarray(cam["t"], dtype=np.float64).copy()
            })

        loss_track = deque(maxlen=history)
        loss_track.append(best_loss)

        wait = 0

        with open(loss_path, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
            f.write(f"-1,{best_loss}\n")
            f.flush()

            for step in range(iters):
                print(f"Iteration {step + 1}/{iters}...")

                global_X_grads, global_t_grads, global_q_grads, global_d_grads = self.backprop()

                old_track_to_point = {
                    root: np.asarray(X, dtype=np.float64).copy()
                    for root, X in self.constructor.track_to_point.items()
                }

                old_cameras = []
                for cam in self.constructor.camera_matrices:
                    old_cameras.append({
                        "k": np.asarray(cam["k"], dtype=np.float64).copy(),
                        "d": np.asarray(cam["d"], dtype=np.float64).copy(),
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
                        cam["q"] = q

                    if img_id in global_d_grads:
                        d = np.asarray(cam["d"], dtype=np.float64).reshape(3)
                        d = d - lr * global_d_grads[img_id]
                        cam["d"] = d

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
                            "k": np.asarray(cam["k"], dtype=np.float64).copy(),
                            "d": np.asarray(cam["d"], dtype=np.float64).copy(),
                            "q": np.asarray(cam["q"], dtype=np.float64).copy(),
                            "t": np.asarray(cam["t"], dtype=np.float64).copy()
                        })
                else:
                    wait += 1
                    print(f"Waiting {wait}/{patience}")

                    if wait >= patience:
                        print(f"Early stopping at iteration {step + 1}")
                        break

        self.constructor.track_to_point = {
            root: np.asarray(X, dtype=np.float64).copy()
            for root, X in best_track_to_point.items()
        }

        for i, cam in enumerate(best_cameras):
            self.constructor.camera_matrices[i]["k"] = cam["k"]
            self.constructor.camera_matrices[i]["d"] = cam["d"]
            self.constructor.camera_matrices[i]["q"] = cam["q"]
            self.constructor.camera_matrices[i]["t"] = cam["t"]

        print("Best reprojection loss:", best_loss)

class LM_optimizer:
    def __init__(self, constructor):
        self.constructor = constructor

    def _quat_norm_jacobian(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too close to zero.")

        q_prime = q / n
        I = np.eye(4, dtype=np.float64)
        J = (I - np.outer(q_prime, q_prime)) / n
        return J

    def _get_tracks(self, save_observation_points=False):
        groups = self.constructor.tracker.groups()
        tracks = []

        for group in groups:
            if len(group) == 0:
                continue

            root = self.constructor.tracker.find(group[0])
            X = self.constructor.track_to_point.get(root)

            if X is None:
                continue

            if save_observation_points:
                observations = []
                for img_id, kp_id in group:
                    features = self.constructor.features[img_id]
                    keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                    observations.append({
                        "camera_id": int(img_id),
                        "point2d": keypoints[int(kp_id)].reshape(2)
                    })
            else:
                observations = group

            tracks.append({
                "root": root,
                "point3d": np.asarray(X, dtype=np.float64).reshape(3),
                "observations": observations
            })

        return tracks

    def _build_K_from_k(self, k):
        fx, fy, cx, cy, s = np.asarray(k, dtype=np.float64).reshape(5)
        return np.array([
            [fx, s,  cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _project_with_radial(self, k, d, Z):
        fx, fy, cx, cy, s = np.asarray(k, dtype=np.float64).reshape(5)
        k1, k2, k3 = np.asarray(d, dtype=np.float64).reshape(3)
        Z0, Z1, Z2 = np.asarray(Z, dtype=np.float64).reshape(3)

        if abs(Z2) < 1e-12:
            return None

        xn = Z0 / Z2
        yn = Z1 / Z2
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
        xd = xn * radial
        yd = yn * radial

        x_hat = np.array([
            fx * xd + s * yd + cx,
            fy * yd + cy
        ], dtype=np.float64)

        return x_hat, xn, yn, r2, radial, xd, yd

    def _pack_theta(self):
        tracks = self._get_tracks()

        point_roots = []
        point_blocks = []

        for track in tracks:
            root = track["root"]
            X = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
            point_roots.append(root)
            point_blocks.append(X)

        cam_blocks = []
        for cam in self.constructor.camera_matrices:
            k = np.asarray(cam["k"], dtype=np.float64).reshape(5)
            d = np.asarray(cam.get("d", self.constructor.radial_parameters), dtype=np.float64).reshape(3)
            q = np.asarray(cam["q"], dtype=np.float64).reshape(4)
            t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
            cam_blocks.append(np.concatenate([k, d, q, t]))

        theta_points = np.concatenate(point_blocks) if len(point_blocks) > 0 else np.zeros(0, dtype=np.float64)
        theta_cams = np.concatenate(cam_blocks) if len(cam_blocks) > 0 else np.zeros(0, dtype=np.float64)
        theta = np.concatenate([theta_points, theta_cams])

        meta = {
            "tracks": tracks,
            "point_roots": point_roots,
            "root_to_point_idx": {root: i for i, root in enumerate(point_roots)},
            "N": len(point_roots),
            "M": len(self.constructor.camera_matrices)
        }

        return theta.astype(np.float64), meta

    def _unpack_theta(self, theta, meta):
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)

        N = meta["N"]
        M = meta["M"]

        expected = 3 * N + 15 * M
        if theta.size != expected:
            raise ValueError(f"Theta size mismatch. Expected {expected}, got {theta.size}")

        point_map = {}
        cursor = 0

        for root in meta["point_roots"]:
            point_map[root] = theta[cursor:cursor + 3].copy()
            cursor += 3

        cameras = []
        for _ in range(M):
            k = theta[cursor:cursor + 5].copy()
            cursor += 5

            d = theta[cursor:cursor + 3].copy()
            cursor += 3

            q = theta[cursor:cursor + 4].copy()
            cursor += 4

            t = theta[cursor:cursor + 3].copy()
            cursor += 3

            cameras.append({
                "k": k,
                "d": d,
                "q": q,
                "t": t
            })

        return point_map, cameras

    def _apply_theta(self, theta, meta):
        point_map, cameras = self._unpack_theta(theta, meta)

        for root, X in point_map.items():
            current_root = self.constructor.tracker.find(root)
            if current_root in self.constructor.track_to_point:
                self.constructor.track_to_point[current_root] = X.copy()

        for i, cam in enumerate(cameras):
            self.constructor.camera_matrices[i]["k"] = cam["k"].copy()
            self.constructor.camera_matrices[i]["d"] = cam["d"].copy()
            self.constructor.camera_matrices[i]["q"] = cam["q"].copy()  # raw q
            self.constructor.camera_matrices[i]["t"] = cam["t"].copy()

    def _dR_dq_prime(self, q_prime):
        w, x, y, z = q_prime

        dR_dw = np.array([
            [0.0, -2.0 * z,  2.0 * y],
            [2.0 * z,  0.0, -2.0 * x],
            [-2.0 * y, 2.0 * x, 0.0]
        ], dtype=np.float64)

        dR_dx = np.array([
            [0.0, 2.0 * y, 2.0 * z],
            [2.0 * y, -4.0 * x, -2.0 * w],
            [2.0 * z,  2.0 * w, -4.0 * x]
        ], dtype=np.float64)

        dR_dy = np.array([
            [-4.0 * y, 2.0 * x,  2.0 * w],
            [2.0 * x,  0.0,      2.0 * z],
            [-2.0 * w, 2.0 * z, -4.0 * y]
        ], dtype=np.float64)

        dR_dz = np.array([
            [-4.0 * z, -2.0 * w, 2.0 * x],
            [2.0 * w,  -4.0 * z, 2.0 * y],
            [2.0 * x,   2.0 * y, 0.0]
        ], dtype=np.float64)

        return dR_dw, dR_dx, dR_dy, dR_dz

    def _build_system(self, theta, meta):
        point_map, cameras = self._unpack_theta(theta, meta)

        N = meta["N"]
        M = meta["M"]
        total_params = 3 * N + 15 * M

        residual_rows = []
        jacobian_rows = []

        for track in meta["tracks"]:
            root = track["root"]
            X = np.asarray(point_map[root], dtype=np.float64).reshape(3)
            observations = track["observations"]

            if observations is None or len(observations) == 0:
                continue

            weight = 1.0 / len(observations)
            sqrt_w = np.sqrt(weight)

            point_idx = meta["root_to_point_idx"][root]
            point_base = 3 * point_idx

            for (img_id, kp_id) in observations:
                features = self.constructor.features[img_id]
                keypoints = features["keypoints"][0].cpu().numpy().astype(np.float64)
                x_ij = keypoints[int(kp_id)].reshape(2)

                cam = cameras[img_id]
                k = np.asarray(cam["k"], dtype=np.float64).reshape(5)
                d = np.asarray(cam["d"], dtype=np.float64).reshape(3)
                q_raw = np.asarray(cam["q"], dtype=np.float64).reshape(4)
                t = np.asarray(cam["t"], dtype=np.float64).reshape(3)

                q_norm = np.linalg.norm(q_raw)
                if q_norm < 1e-12:
                    continue

                q_prime = q_raw / q_norm
                R = utility.quaternion_to_R(q_raw)

                Z = R @ X + t
                projected = self._project_with_radial(k, d, Z)
                if projected is None:
                    continue

                x_hat, xn, yn, r2, radial, xd, yd = projected

                r_ij = sqrt_w * (x_ij - x_hat)

                fx, fy, _, _, s = k
                k1, k2, k3 = d
                Z0, Z1, Z2 = Z

                radial_slope = k1 + 2.0 * k2 * r2 + 3.0 * k3 * (r2 ** 2)
                dradial_dxn = 2.0 * xn * radial_slope
                dradial_dyn = 2.0 * yn * radial_slope

                dxd_dxn = radial + xn * dradial_dxn
                dxd_dyn = xn * dradial_dyn
                dyd_dxn = yn * dradial_dxn
                dyd_dyn = radial + yn * dradial_dyn

                dproj_dnorm = np.array([
                    [fx * dxd_dxn + s * dyd_dxn, fx * dxd_dyn + s * dyd_dyn],
                    [fy * dyd_dxn, fy * dyd_dyn]
                ], dtype=np.float64)

                dnorm_dZ = np.array([
                    [1.0 / Z2, 0.0, -Z0 / (Z2 ** 2)],
                    [0.0, 1.0 / Z2, -Z1 / (Z2 ** 2)]
                ], dtype=np.float64)

                dr_dZ = -sqrt_w * (dproj_dnorm @ dnorm_dZ)
                dr_dX = dr_dZ @ R
                dr_dt = dr_dZ.copy()

                dr_dk = -sqrt_w * np.array([
                    [xd, 0.0, 1.0, 0.0, yd],
                    [0.0, yd, 0.0, 1.0, 0.0]
                ], dtype=np.float64)

                radial_powers = np.array([r2, r2 ** 2, r2 ** 3], dtype=np.float64)
                dr_dd = np.zeros((2, 3), dtype=np.float64)
                for j, power in enumerate(radial_powers):
                    dxd_dd = xn * power
                    dyd_dd = yn * power
                    dr_dd[:, j] = -sqrt_w * np.array([
                        fx * dxd_dd + s * dyd_dd,
                        fy * dyd_dd
                    ], dtype=np.float64)

                dR_dw, dR_dx, dR_dy, dR_dz = self._dR_dq_prime(q_prime)

                dZ_dw = dR_dw @ X
                dZ_dx = dR_dx @ X
                dZ_dy = dR_dy @ X
                dZ_dz = dR_dz @ X

                dr_dq_prime = np.column_stack([
                    dr_dZ @ dZ_dw,
                    dr_dZ @ dZ_dx,
                    dr_dZ @ dZ_dy,
                    dr_dZ @ dZ_dz
                ])

                J_norm = self._quat_norm_jacobian(q_raw)
                dr_dq_raw = dr_dq_prime @ J_norm

                row = np.zeros((2, total_params), dtype=np.float64)

                row[:, point_base:point_base + 3] = dr_dX

                cam_base = 3 * N + 15 * img_id
                row[:, cam_base:cam_base + 5] = dr_dk
                row[:, cam_base + 5:cam_base + 8] = dr_dd
                row[:, cam_base + 8:cam_base + 12] = dr_dq_raw
                row[:, cam_base + 12:cam_base + 15] = dr_dt

                residual_rows.append(r_ij)
                jacobian_rows.append(row)

        if len(residual_rows) == 0:
            return np.zeros(0, dtype=np.float64), np.zeros((0, total_params), dtype=np.float64)

        r = np.vstack(residual_rows).reshape(-1)
        J = np.vstack(jacobian_rows)

        return r, J

    def _normalized_loss(self, r):
        if r.size == 0:
            return 0.0
        return np.sqrt(np.dot(r, r) / r.size)

    def optimize(self, iters=20, lambda_init=1e-3, lambda_scale=10.0,
                 step_tol=1e-8, loss_tol=1e-12, loss_file="lm_loss_log.txt"):
        loss_path = Path(loss_file)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        theta, meta = self._pack_theta()

        best_theta = theta.copy()
        best_r, _ = self._build_system(theta, meta)
        best_loss = 0.5 * np.dot(best_r, best_r) if best_r.size > 0 else 0.0

        lambda_ = float(lambda_init)

        with open(loss_path, "w", encoding="utf-8") as f:
            f.write("step,loss,lambda\n")
            f.write(f"0,{self._normalized_loss(best_r)},{lambda_}\n")
            f.flush()

            for step in range(iters):
                print(f"LM Iteration {step + 1}/{iters}...")

                r, J = self._build_system(theta, meta)
                current_loss = 0.5 * np.dot(r, r) if r.size > 0 else 0.0
                current_loss_norm = self._normalized_loss(r)

                if r.size == 0:
                    print("No valid residuals.")
                    break

                A = J.T @ J + lambda_ * np.eye(theta.size, dtype=np.float64)
                g = J.T @ r

                try:
                    delta = np.linalg.solve(A, -g)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(A, -g, rcond=None)[0]

                delta_norm = np.linalg.norm(delta)
                if delta_norm < step_tol:
                    print("LM stopped: step norm is small.")
                    break

                theta_candidate = theta + delta

                point_map_candidate, cameras_candidate = self._unpack_theta(theta_candidate, meta)

                valid = True
                for cam in cameras_candidate:
                    k = cam["k"]
                    q = cam["q"]

                    if np.linalg.norm(q) < 1e-12:
                        valid = False
                        break
                    if k[0] <= 1e-9 or k[1] <= 1e-9:
                        valid = False
                        break

                if not valid:
                    lambda_ *= lambda_scale
                    print(f"Rejected (invalid params). lambda -> {lambda_:.6e}")
                    f.write(f"{step},{current_loss_norm},{lambda_}\n")
                    f.flush()
                    continue

                r_candidate, _ = self._build_system(theta_candidate, meta)
                candidate_loss = 0.5 * np.dot(r_candidate, r_candidate) if r_candidate.size > 0 else current_loss
                candidate_loss_norm = self._normalized_loss(r_candidate)

                if candidate_loss < current_loss:
                    theta = theta_candidate
                    lambda_ /= lambda_scale

                    print(f"Accepted. RMSE: {current_loss_norm:.12f} -> {candidate_loss_norm:.12f}")

                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_theta = theta.copy()

                    if abs(current_loss - candidate_loss) < loss_tol:
                        print("LM stopped: loss improvement is small.")
                        f.write(f"{step},{candidate_loss_norm},{lambda_}\n")
                        f.flush()
                        break
                else:
                    lambda_ *= lambda_scale
                    print(f"Rejected. RMSE: {candidate_loss_norm:.12f}, lambda -> {lambda_:.6e}")

                log_loss = candidate_loss_norm if candidate_loss < current_loss else current_loss_norm
                f.write(f"{step},{log_loss},{lambda_}\n")
                f.flush()

        self._apply_theta(best_theta, meta)

        best_r, _ = self._build_system(best_theta, meta)
        print("Best LM RMSE:", self._normalized_loss(best_r))

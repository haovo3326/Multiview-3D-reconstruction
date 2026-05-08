from itertools import combinations
from pathlib import Path
from optimizer import load_tracks
import utility
import numpy as np
import matplotlib.pyplot as plt


SAVE_DIR = Path("save")
LOSS_FILE = SAVE_DIR / "loss_log.txt"
TRACKS_FILE = SAVE_DIR / "tracks.pkl"
CAMERAS_FILE = SAVE_DIR / "cameras.json"
CLOUD_FILE = SAVE_DIR / "output.ply"
FILTERED_CLOUD_FILE = SAVE_DIR / "output_filtered.ply"
REPROJECTION_ACCEPTANCE_RATE = 0.6
ANGLE_COSINE_ACCEPTANCE_RATE = 0.9
BIN_COUNT = 1000

def collect_reprojection_error(tracks, camera_matrices):
    reprojection_error = []
    reprojection_track_indices = []

    for track_index, track in enumerate(tracks):
        point3d = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
        total_track_reprojection_error = 0.0
        valid_projection_count = 0

        for sample in track["observations"]:
            camera_id = sample["camera_id"]
            camera = camera_matrices[camera_id]
            point2d = np.asarray(sample["point2d"], dtype=np.float64).reshape(2)

            projected_point2d = utility.project_point_with_radial(
                camera["k"],
                camera["d"],
                camera["q"],
                camera["t"],
                point3d
            )

            if projected_point2d is None:
                continue

            total_track_reprojection_error += np.linalg.norm(point2d - projected_point2d)
            valid_projection_count += 1

        if valid_projection_count == 0:
            continue

        reprojection_error.append(total_track_reprojection_error / valid_projection_count)
        reprojection_track_indices.append(track_index)

    return (
        np.asarray(reprojection_error, dtype=np.float64),
        np.asarray(reprojection_track_indices, dtype=np.int64)
    )

def collect_angle_cosines(tracks, camera_matrices, track_mask=None):
    camera_centers = []
    for camera in camera_matrices:
        R_mat = utility.quaternion_to_R(camera["q"])
        t = np.asarray(camera["t"], dtype=np.float64).reshape(3)
        camera_centers.append(-R_mat.T @ t)

    if track_mask is None:
        track_mask = np.ones(len(tracks), dtype=bool)

    angle_cosines = []
    angle_track_indices = []
    for track_index, (track, is_valid_track) in enumerate(zip(tracks, track_mask)):
        if not is_valid_track:
            continue

        if len(track["observations"]) < 2:
            continue

        point3d = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
        observed_camera_ids = [
            sample["camera_id"]
            for sample in track["observations"]
        ]
        observed_camera_ids = sorted(set(observed_camera_ids))

        track_cosines = []
        for camera_id_a, camera_id_b in combinations(observed_camera_ids, 2):
            vector_a = camera_centers[camera_id_a] - point3d
            vector_b = camera_centers[camera_id_b] - point3d
            norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

            if norm_product < 1e-12:
                continue

            cosine = np.dot(vector_a, vector_b) / norm_product
            track_cosines.append(np.clip(cosine, -1.0, 1.0))

        if len(track_cosines) == 0:
            continue

        angle_cosines.append(np.mean(track_cosines))
        angle_track_indices.append(track_index)

    return (
        np.asarray(angle_cosines, dtype=np.float64),
        np.asarray(angle_track_indices, dtype=np.int64)
    )

def build_statistic(data):
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("No finite data points to build histogram statistic.")

    min_err = np.min(data)
    max_err = np.max(data)

    counts, bin_edges = np.histogram(data, bins=BIN_COUNT)

    return {
        "min": min_err,
        "max": max_err,
        "mean": np.mean(data),
        "median": np.median(data),
        "counts": counts,
        "bin_edges": bin_edges
    }

def threshold_from_histogram(stat, acceptance_rate):
    counts = np.asarray(stat["counts"], dtype=np.float64)
    bin_edges = stat["bin_edges"]

    total = np.sum(counts)
    if total == 0:
        raise ValueError("Empty histogram")

    # CDF
    cdf = np.cumsum(counts) / total

    # find bin index
    idx = np.searchsorted(cdf, acceptance_rate, side="left")
    idx = min(idx, len(bin_edges) - 2)

    # threshold = right edge
    threshold = bin_edges[idx + 1]
    stat["acceptance_rate"] = acceptance_rate
    stat["threshold"] = threshold
    return threshold

def build_track_mask(track_count, track_indices, values, threshold, keep_less_equal=True):
    mask = np.zeros(track_count, dtype=bool)
    track_indices = np.asarray(track_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if keep_less_equal:
        accepted_indices = track_indices[values <= threshold]
    else:
        accepted_indices = track_indices[values >= threshold]

    mask[accepted_indices] = True
    return mask

def combine_reprojection_and_angle_masks(
        track_count,
        reprojection_track_indices,
        reprojection_error,
        reprojection_error_threshold,
        angle_track_indices,
        angle_cosine,
        angle_cosine_threshold):
    reprojection_mask = build_track_mask(
        track_count,
        reprojection_track_indices,
        reprojection_error,
        reprojection_error_threshold,
        keep_less_equal=True
    )
    angle_cosine_mask = build_track_mask(
        track_count,
        angle_track_indices,
        angle_cosine,
        angle_cosine_threshold,
        keep_less_equal=True
    )

    return reprojection_mask & angle_cosine_mask

def mask_to_points_and_colors(tracks, mask):
    points = []
    colors = []
    has_color = False

    for track, is_valid_track in zip(tracks, mask):
        if not is_valid_track:
            continue

        point3d = np.asarray(track["point3d"], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point3d)):
            continue

        points.append(point3d)

        color = track.get("color")
        if color is None:
            colors.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        else:
            color = np.clip(np.asarray(color, dtype=np.float64).reshape(3), 0.0, 1.0)
            colors.append(color)
            has_color = True

    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), None

    colors = np.asarray(colors, dtype=np.float64) if has_color else None
    return np.asarray(points, dtype=np.float64), colors

def load_ply_colors_for_points(points, filename):
    path = Path(filename)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vertex_count = 0
    header_end = None
    has_color = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
        elif stripped == "property uchar red":
            has_color = True
        elif stripped == "end_header":
            header_end = i + 1
            break

    if header_end is None or not has_color:
        return None

    color_by_point = {}
    for line in lines[header_end:header_end + vertex_count]:
        values = line.split()
        if len(values) < 6:
            continue
        point = np.asarray(values[:3], dtype=np.float64)
        color = np.asarray(values[3:6], dtype=np.float64) / 255.0
        color_by_point[tuple(np.round(point, 12))] = color

    colors = []
    for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        color = color_by_point.get(tuple(np.round(point, 12)))
        if color is None:
            return None
        colors.append(color)

    return np.asarray(colors, dtype=np.float64)

def plot_histogram(histogram, threshold, title, xlabel):
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    bin_edges = np.asarray(histogram["bin_edges"], dtype=np.float64)
    bin_widths = np.diff(bin_edges)
    min_value = histogram["min"]
    max_value = histogram["max"]
    annotation = (
        f"min = {min_value:.6f}    "
        f"max = {max_value:.6f}    "
        f"threshold = {threshold:.6f}"
    )

    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        counts,
        width=bin_widths,
        align="edge",
        edgecolor="black"
    )
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="threshold"
    )
    ax.axvline(
        min_value,
        color="green",
        linestyle=":",
        linewidth=2.0,
        label="min"
    )
    ax.axvline(
        max_value,
        color="blue",
        linestyle=":",
        linewidth=2.0,
        label="max"
    )
    ax.text(
        0.5,
        1.08,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="bottom"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_point_cloud(points, colors=None, title="Filtered Point Cloud"):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = None if colors is None else np.asarray(colors, dtype=np.float64).reshape(-1, 3)

    valid_mask = np.isfinite(points).all(axis=1)
    if colors is not None:
        valid_mask &= np.isfinite(colors).all(axis=1)

    points = points[valid_mask]
    if colors is not None:
        colors = np.clip(colors[valid_mask], 0.0, 1.0)

    if points.shape[0] == 0:
        print("No valid points to plot.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    plt.tight_layout()
    plt.show()

def main():
    if not TRACKS_FILE.exists():
        raise FileNotFoundError(
            f"{TRACKS_FILE} does not exist. Run main.py once to save tracks first."
        )
    if not CAMERAS_FILE.exists():
        raise FileNotFoundError(
            f"{CAMERAS_FILE} does not exist. Run main.py once to save cameras first."
        )

    tracks = load_tracks(TRACKS_FILE)
    camera_matrices = utility.load_camera_info(CAMERAS_FILE)

    reprojection_error, reprojection_track_indices = collect_reprojection_error(tracks, camera_matrices)
    reprojection_error_histogram = build_statistic(reprojection_error)

    reprojection_error_threshold = threshold_from_histogram(
        reprojection_error_histogram,
        REPROJECTION_ACCEPTANCE_RATE
    )
    reprojection_mask = build_track_mask(
        len(tracks),
        reprojection_track_indices,
        reprojection_error,
        reprojection_error_threshold,
        keep_less_equal=True
    )

    angle_cosine, angle_track_indices = collect_angle_cosines(
        tracks,
        camera_matrices,
        reprojection_mask
    )
    angle_cosine_histogram = build_statistic(angle_cosine)

    angle_cosine_threshold = threshold_from_histogram(
        angle_cosine_histogram,
        ANGLE_COSINE_ACCEPTANCE_RATE
    )
    final_mask = combine_reprojection_and_angle_masks(
        len(tracks),
        reprojection_track_indices,
        reprojection_error,
        reprojection_error_threshold,
        angle_track_indices,
        angle_cosine,
        angle_cosine_threshold
    )
    filtered_points, filtered_colors = mask_to_points_and_colors(tracks, final_mask)
    if filtered_colors is None:
        filtered_colors = load_ply_colors_for_points(filtered_points, CLOUD_FILE)

    utility.save_point_cloud_ply(filtered_points, FILTERED_CLOUD_FILE, filtered_colors)

    plot_histogram(
        reprojection_error_histogram,
        reprojection_error_threshold,
        "Reprojection Error Histogram",
        "Reprojection Error"
    )
    plot_histogram(
        angle_cosine_histogram,
        angle_cosine_threshold,
        "Angle Cosine Histogram",
        "Angle Cosine"
    )
    plot_point_cloud(filtered_points, filtered_colors)


if __name__ == "__main__":
    main()

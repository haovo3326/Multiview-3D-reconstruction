import cv2
import numpy as np
import torch
import kornia.feature as KF

def get_correspondence(img1Gray, img2Gray):
    # LoFTR expects grayscale images
    gray1 = cv2.cvtColor(img1Gray, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2Gray, cv2.COLOR_BGR2GRAY)
    # Resize
    scale = 1
    gray1_small = cv2.resize(gray1, None, fx = scale, fy = scale)
    gray2_small = cv2.resize(gray2, None, fx=scale, fy=scale)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor with shape [1, 1, H, W]
    timg1 = torch.from_numpy(gray1_small).float() / 255.0
    timg2 = torch.from_numpy(gray2_small).float() / 255.0

    timg1 = timg1.unsqueeze(0).unsqueeze(0).to(device)
    timg2 = timg2.unsqueeze(0).unsqueeze(0).to(device)

    # Load pretrained LoFTR
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    with torch.no_grad():
        correspondences = matcher({
            "image0": timg1,
            "image1": timg2
        })

    # Extract matched points
    pts1 = correspondences["keypoints0"].cpu().numpy().astype(np.float32)
    pts2 = correspondences["keypoints1"].cpu().numpy().astype(np.float32)
    conf = correspondences["confidence"].cpu().numpy()

    # Keep only stronger matches
    mask = conf > 0.3
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    conf = conf[mask]

    # Keep only top-N
    max_matches = 1000
    idx = np.argsort(-conf)[:max_matches]
    pts1 = pts1[idx]
    pts2 = pts2[idx]

    # scale points back to original image size
    pts1[:, 0] /= scale
    pts1[:, 1] /= scale
    pts2[:, 0] /= scale
    pts2[:, 1] /= scale

    return pts1, pts2


def rounding_and_unique(K, pts1, pts2):
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    pts1_inlier = pts1[mask.ravel() == 1]
    pts2_inlier = pts2[mask.ravel() == 1]

    pts1_int = [(round(x), round(y)) for x, y in pts1_inlier]
    pts2_int = [(round(x), round(y)) for x, y in pts2_inlier]

    pairs = []

    K_inv = np.linalg.inv(K)
    for p1, p2 in zip(pts1_int, pts2_int):
        x1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
        x2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
        x1n = K_inv @ x1
        x2n = K_inv @ x2
        error = abs(x2n @ E @ x1n)
        pairs.append((p1, p2, error))

    pairs.sort(key=lambda t: t[2])

    used1 = set()
    used2 = set()
    best_pairs = []

    for p1, p2, error in pairs:
        if p1 not in used1 and p2 not in used2:
            best_pairs.append((p1, p2))
            used1.add(p1)
            used2.add(p2)

    pts1_unique = np.array([p1 for p1, p2 in best_pairs], dtype=np.float32)
    pts2_unique = np.array([p2 for p1, p2 in best_pairs], dtype=np.float32)

    _, R, t, pose_mask = cv2.recoverPose(E, pts1_unique, pts2_unique, K)

    pts1_final = pts1_unique[pose_mask.ravel() == 255]
    pts2_final = pts2_unique[pose_mask.ravel() == 255]

    return R, t, pts1_final, pts2_final

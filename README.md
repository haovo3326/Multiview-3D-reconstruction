# A demonstration of complete 3D reconstruction pipeline utilizing SuperPoint + SuperGlue and PnP (Perspective-n-Point)
## Overview
This project demonstrates the pipeline for reconstructing 3D scene from images captured by the same camera from multiple angles. The pipeline includes 3 stages:
1. Camera calibration
2. Feature detection and matching
3. Feature tracking
4. Triangulation

## Camera calibration
This stage specifically focus on obtaining the intrinsic matrix K of the camera by taking multiple images of checkerboard pattern and apply Zhang's calibration algorithm.

## Feature detection and matching
For feature detection, SuperPoint model is used to detect features (keypoints and descriptors) of individual images. After that, SuperGlue is included in identify correspondances between image pairs. Both models are taken from the original repository of LightGlue https://github.com/cvg/lightglue

## Feature tracking
In this stage, Disjoint Set Union (DSU) structure is required to track similar features across multiple images. Any long track is used for camera pose computation in PnP (Perspective-n-Point) and avoid recomputation of existing 3D features.

## Triangulation
This step involves the recovery of 3D coordinates from the 2D features tracks.

## Dependencies
The project uses the following libraries
- Python 3.9+
- OpenCV
- Numpy
- Pytorch
- LightGlue (Repo)
- Matplotlib




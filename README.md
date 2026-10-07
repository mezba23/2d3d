# 2D to 3D Vision Model Reconstruction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Pose%20Estimation-purple.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision pipeline that reconstructs **3D spatial meshes and point clouds** from single/multi-view **2D RGB images** utilizing human pose estimation and monocular depth cues.

---

## Technical Pipeline

```
  2D Input Image
        │
        ├──▶ [ Pose Estimation ] ──▶ 2D Joint Landmarks (X, Y)
        │                                      │
        └──▶ [ Depth Estimation ] ─▶ Depth Map (Z)
                                               │
                                               ▼
                             [ 3D Euclidean Lifting ]
                                               │
                                               ▼
                                 Exported Wavefront (.obj)
```

---

## Features

- **2D Landmark Extraction**: Detects skeletal joints (shoulders, elbows, hips, knees).
- **Depth Map Generation**: Estimates depth gradients across visual scenes.
- **3D Spatial Lifting**: Merges (X, Y) pixel coordinates with depth (Z) to produce 3D Euclidean coordinates.
- **Wavefront .obj Export**: Exports generated meshes for direct viewing in Blender or 3D viewer engines.

---

## Quickstart

```bash
pip install -r requirements.txt
python reconstruct_pipeline.py
```

Inspect the generated `output_reconstruction.obj` in any 3D model viewer!

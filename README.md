# 6DOF IK & 3D Reconstruction

This project combines two complementary capabilities:

- Inverse Kinematics (IK) solver for a 6-DOF ABB IRB 6700 mounted on a linear gantry.
- 3D Pose Estimation of objects using depth and color data from a camera (e.g., Intel RealSense or similar).

Together, they enable detecting an object in 3D space and computing joint configurations to reach it.

---

## Features

### 6-DOF Gantry IK
- Extracts **Modified DH parameters** from a URDF robot model.
- Supports:
  - **Analytical IK** (8–16 configurations)
  - **Numerical IK** (Damped Least Squares)
- Includes forward kinematics, Jacobian computation, and validation.

### 3D Pose Estimation
- Processes RGB-D frames to detect foreground objects using a **depth gradient-based segmentation** approach.
- Extracts box pose using **PCA** and **RANSAC** plane fitting.
- Computes pose in both camera and world frames.
- Visualizes point clouds and poses using Open3D.

---

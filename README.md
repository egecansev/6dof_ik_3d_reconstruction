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
  - **Analytical IK**: Computes multiple (up to 16) joint angle configurations for a given end-effector
  pose using a closed-form solution based on modified Denavit-Hartenberg parameters.
  - **Numerical IK**Uses a damped least squares iterative method to find a joint solution starting from an initial guess.
- Includes forward kinematics, Jacobian computation, and validation.

### Note on Analytical IK Solver Accuracy

The analytical IK solver currently produces multiple candidate solutions; however, only a subset of these solutions are
valid and satisfy zero position and orientation error. Some solutions may be physically infeasible or yield significant
errors. Therefore, this solver is still a **work in progress** and requires further improvements to enhance robustness
and completeness before being used in critical applications.

The numerical IK solver can be used as a fallback or refinement method to ensure convergence to valid joint configurations.


### 3D Pose Estimation
- Processes RGB-D frames to detect foreground objects using a **depth gradient-based segmentation** approach.
- Extracts box pose using **PCA** and **RANSAC** plane fitting.
- Computes pose in both camera and world frames.
- Visualizes point clouds and poses using Open3D.

---

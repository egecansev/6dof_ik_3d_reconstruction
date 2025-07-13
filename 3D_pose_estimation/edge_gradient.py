import numpy as np
import open3d as o3d
import time
import cv2
from scipy.spatial.transform import Rotation


extrinsics = np.load("extrinsics.npy")
intrinsics = np.load("intrinsics.npy")
color = np.load("one-box.color.npdata.npy")
depth = np.load("one-box.depth.npdata.npy")
extrinsics[:3, 3] /= 1000.0

def get_3d_points(d, ints):
    h, w = d.shape
    fx, fy = ints[0, 0], ints[1, 1]
    cx, cy = ints[0, 2], ints[1, 2]
    z = d.astype(np.float32)
    rows = np.arange(h).reshape(-1, 1)
    cols = np.arange(w).reshape(1, -1)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.empty((h, w, 3), dtype=np.float32)
    points[..., 0] = x
    points[..., 1] = y
    points[..., 2] = z
    return points

def extract_foreground_mask(d, percentile=5):
    valid = (d > 0)
    threshold = np.percentile(d[valid], percentile)
    return (d > 0) & (d < threshold)

def intensity_filter_mask(gray_img, lower=100, upper=200):
    return (gray_img >= lower) & (gray_img <= upper)

def to_open3d_cloud(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p

def create_axis_lines(center, rotation, length=0.2):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = []
    for i in range(3):
        axis = o3d.geometry.LineSet()
        pts = [center, center + rotation[:, i] * length]
        axis.points = o3d.utility.Vector3dVector(pts)
        axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        axis.colors = o3d.utility.Vector3dVector([colors[i]])
        axes.append(axis)
    return axes

def run_pca(points):
    cent = np.median(points, axis=0)
    centered = points - cent
    cov = np.cov(centered, rowvar=False)
    ei_val, ei_vec = np.linalg.eigh(cov)
    idx = np.argsort(ei_val)[::-1]
    return ei_val[idx], ei_vec[:, idx], cent, cov

def project_points_to_image(p3d, ints):
    fx, fy = ints[0, 0], ints[1, 1]
    cx, cy = ints[0, 2], ints[1, 2]
    x, y, z = p3d[:, 0], p3d[:, 1], p3d[:, 2]
    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)
    return np.stack([u, v], axis=1)


def pose_to_euler_and_translation(T, degrees=True):
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    r = Rotation.from_matrix(R_mat)
    euler = r.as_euler('xyz', degrees=degrees)
    quat = r.as_quat()  # (x, y, z, w) format
    return t_vec, euler, quat


gray = color.astype(np.uint8)
points_3d = get_3d_points(depth, intrinsics)

valid_mask = depth > 0
full_points = points_3d[valid_mask]
pcd_full = to_open3d_cloud(full_points)

foreground_mask = extract_foreground_mask(depth)
intensity_mask = intensity_filter_mask(gray)
combined_mask = foreground_mask & intensity_mask
filtered_points = points_3d[combined_mask]

pcd = to_open3d_cloud(filtered_points)
pcd_clean, ind = pcd.remove_statistical_outlier(20, 2.0)
# RANSAC with 100.000 iterations are better, but much costly
start_time = time.time()
plane_model, inliers = pcd_clean.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=10000)
print(f"Plane segmentation took {time.time() - start_time:.3f} seconds.")
pcd_clean = pcd_clean.select_by_index(inliers, invert=True)

box_points = np.asarray(pcd_clean.points)
median_box_height = np.median(box_points[:, 2])

# Estimate pallet height
box_pixels = project_points_to_image(box_points, intrinsics)
box_pixels = np.clip(box_pixels, [0, 0], [depth.shape[1]-1, depth.shape[0]-1])

if len(box_pixels) >= 3:
    hull = cv2.convexHull(box_pixels)
    box_mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.fillConvexPoly(box_mask, hull, 1)
else:
    box_mask = np.zeros_like(depth, dtype=np.uint8)

kernel = np.ones((15, 15), np.uint8)
dilated = cv2.dilate(box_mask, kernel, iterations=1)
ring_mask = (dilated > 0) & (box_mask == 0)

depth_float = np.where(depth == 0, np.nan, depth.astype(np.float32))
sobel_x = cv2.Sobel(depth_float, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(depth_float, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
edge_mask = gradient_magnitude > 0.005

pallet_sample_mask = ring_mask & edge_mask
pallet_ring_depths = depth[pallet_sample_mask]
pallet_ring_depths = pallet_ring_depths[pallet_ring_depths > 0]
median_pallet_height = np.median(pallet_ring_depths) if len(pallet_ring_depths) else np.median(depth[depth > 0])
depth_diff = median_pallet_height - median_box_height

# PCA and pose estimation
eigenvalues, eigenvectors, centroid, covariance = run_pca(box_points)
centroid[2] += depth_diff / 2

obb = o3d.geometry.OrientedBoundingBox()
obb.center = centroid
obb.R = eigenvectors
raw_extent = np.ptp(box_points @ eigenvectors, axis=0)
extent = raw_extent.copy()
extent[2] = abs(depth_diff)
obb.extent = extent


T_camera_to_box = np.eye(4)
T_camera_to_box[:3, :3] = obb.R
T_camera_to_box[:3, 3] = obb.center
t_cam, euler_cam, quat_cam = pose_to_euler_and_translation(T_camera_to_box)
T_world_to_box = extrinsics @ T_camera_to_box
t_world, euler_world, quat_world = pose_to_euler_and_translation(T_world_to_box)

print("\n--- Box Pose in Camera Frame ---")
print("Estimated 4x4 pose matrix (camera to box):\n", np.array2string(T_camera_to_box, precision=3, suppress_small=True,
                                                                      floatmode='fixed'))
print("Translation (m):", np.array2string(t_cam, precision=3, suppress_small=True, floatmode='fixed'))
print("Euler angles (deg):", np.array2string(euler_cam, precision=3, suppress_small=True, floatmode='fixed'))
print("Quaternion (x, y, z, w):", np.array2string(quat_cam, precision=3, suppress_small=True, floatmode='fixed'))

print("\n--- Box Pose in World Frame ---")
print("Estimated 4x4 pose matrix (world to box):\n", np.array2string(T_world_to_box, precision=3, suppress_small=True,
                                                                     floatmode='fixed'))
print("Translation (m):", np.array2string(t_world, precision=3, suppress_small=True, floatmode='fixed'))
print("Euler angles (deg):", np.array2string(euler_world, precision=3, suppress_small=True, floatmode='fixed'))
print("Quaternion (x, y, z, w):", np.array2string(quat_world, precision=3, suppress_small=True, floatmode='fixed'))



axes_lines = create_axis_lines(obb.center, obb.R)
center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
center_marker.paint_uniform_color([0, 0, 0])
center_marker.translate(obb.center)


vis = o3d.visualization.Visualizer()
vis.create_window()
for geom in [pcd_full, obb, center_marker, *axes_lines]:
    vis.add_geometry(geom)

ctr = vis.get_view_control()
# Set camera parameters to look from top
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])
ctr.set_lookat(obb.center)

vis.run()
vis.destroy_window()


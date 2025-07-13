from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_transform(pose):
    x, y, z = pose[:3]
    roll, pitch, yaw = pose[-3:]

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

def get_rot_mat(q, theta, alpha):
    theta += q

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    r = np.array([[ct, -st, 0],
                 [ca * st, ca * ct, -sa],
                 [sa * st, sa * ct, ca]])

    return r

# Wrap the angles to (-pi, pi]
def wrap_to_pi(angle):
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if not np.isclose(wrapped, -np.pi):
        return wrapped
    else:
        return np.pi


def extract_modified_dh(robot):
    dh_table = []

    joints = [j for j in robot.joints if j.joint_type == 'revolute']
    joint_names = [j.name for j in joints]

    # Store joint positions and z-axes for x-axis computation
    joint_origins = []
    z_axes = []

    for joint in joints:
        T = joint.origin
        p = T[:3, 3]
        R_joint = T[:3, :3]
        z_axis = R_joint @ np.array(joint.axis)

        joint_origins.append(p)
        if not z_axes:
            z_axes.append(z_axis)   # Base frame z_0
        z_axes.append(z_axis)

    z_axes.append(z_axis)   # Flange frame z_f

    x_axes = []
    for i in range(len(z_axes) - 1):
        z_curr = z_axes[i]
        z_next = z_axes[i + 1]
        z_cross = np.cross(z_curr, z_next)
        if np.linalg.norm(z_cross):
            x_i = z_cross
        else:
            if z_curr[2]:
                x_i = np.array([1.0, 0.0, 0.0])
            else:
                x_i = np.array([0.0, 0.0, 1.0])
        x_i = x_i / np.linalg.norm(x_i)
        if np.dot(x_i, np.ones(3)) < 0:
            x_i *= -1   # let all x be positive unit vectors for convenience
        x_axes.append(x_i)

    x_axes.append(x_i)  # Flange frame z_f

    ## Manipulating joint origins for IK solution

    # Move d4 to previous transition so that RF4 and RF5 have the same origin
    joint_origins[3] += joint_origins[4]
    joint_origins[4] -= joint_origins[4]

    # Move d6 to next transition so that RF5 and RF6 have the same origin
    joint_origins.insert(5, np.zeros(3))


    for i in range(len(z_axes) - 1):
        z_curr = z_axes[i]
        z_next = z_axes[i + 1]
        x_curr = x_axes[i]
        x_next = x_axes[i + 1]

        z_cross = np.cross(z_curr, z_next)
        x_cross = np.cross(x_curr, x_next)

        sin_alpha = np.dot(z_cross, x_curr)
        cos_alpha = np.dot(z_curr, z_next)
        alpha = np.arctan2(sin_alpha, cos_alpha)

        sin_theta = np.dot(x_cross, z_next)
        cos_theta = np.dot(x_curr, x_next)
        theta = np.arctan2(sin_theta, cos_theta)

        p_curr = joint_origins[i]
        a = np.dot(p_curr, x_curr)
        d = np.dot(p_curr, z_next)


        dh_table.append({
            'joint': joint_names[i] if i < len(joint_names) else 'fixed/flange',
            'theta': theta,
            'd': d,
            'a': a,
            'alpha': alpha,
            'x_axis': x_curr.round(3).tolist(),
            'z_axis': z_curr.round(3).tolist()
        })

    return dh_table


def inverse_kinematics(T_0f, dh_table):
    # Extract end-effector position and orientation
    p_e = T_0f[:3, 3]
    R_0f = T_0f[:3, :3]

    # a and d from dh_table
    d1 = dh_table[0]['d']
    d4 = dh_table[3]['d']
    d7 = dh_table[6]['d']

    a2 = dh_table[1]['a']
    a3 = dh_table[2]['a']
    a4 = dh_table[3]['a']

    solutions = []

    # Compute wrist center
    z_axis = R_0f[:, 2]
    p_w = p_e - d7 * z_axis
    for signA in [-1, 1]:

        A = signA * np.sqrt(p_w[0]**2 + p_w[1]**2) - a2
        B = p_w[2] - d1

        C = (A**2 + B**2 - a3**2 - a4**2 - d4**2) / (2 * a3)

        # Use the cosine summation trick
        beta = np.arctan2(d4, a4)

        c3beta = C / np.sqrt(a4**2 + d4**2)

        if abs(c3beta) > 1.0:
            print("[IK ERROR] Pose is outside reachable workspace.")
            continue

        for sign1 in [-1, 1]: # Shoulder left, shoulder right
            q1 = np.arctan2(sign1 * p_w[1], sign1 * p_w[0])

            for sign3 in [-1, 1]: # Elbow up, elbow down
                s3beta = sign3 * np.sqrt(1 - c3beta**2)
                q3 = np.arctan2(s3beta, c3beta) - beta

                # Manipulate A and B to isolate s2 and c2
                M = a3 + a4 * np.cos(q3) - d4 * np.sin(q3)
                N = a4 * np.sin(q3) + d4 * np.cos(q3)

                q2 = np.arctan2(A * M - B * N, A * N + B * M)



                solution = [q1, q2, q3]

                R_03 = np.eye(3)
                for i, q in enumerate(solution):
                    R_03 = R_03 @ get_rot_mat(q, dh_table[i]["theta"], dh_table[i]["alpha"])

                R_3f = R_03.T @ R_0f



                for sign5 in [-1, 1]: # Wrist not flipped, wrist flipped
                    q5 = np.arctan2(sign5 * np.sqrt(R_3f[0, 2]**2 + R_3f[2, 2]**2), R_3f[1, 2])
                    q4 = np.arctan2(R_3f[2, 2], -R_3f[0, 2])
                    q6 = np.arctan2(-R_3f[1, 1], R_3f[1, 0])
                    wrapped = np.array([wrap_to_pi(q) for q in [q1, q2, q3, q4, q5, q6]])
                    solutions.append(wrapped)

    return solutions


def forward_kinematics(joint_angles, dh_table):
    T = np.eye(4)
    joint_angles = np.append(joint_angles, 0)  # Fixed flange link
    for i, q in enumerate(joint_angles):
        theta = dh_table[i]["theta"] + q
        d = dh_table[i]["d"]
        a = dh_table[i]["a"]
        alpha = dh_table[i]["alpha"]

        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T_i = np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0, 0, 0, 1]
        ])
        T = T @ T_i
    return T


def check_ik_solutions(solutions, Td, dh_table, tol=1e-4):
    valid_count = 0
    valid_configs = []
    for idx, sol in enumerate(solutions):
        T_fk = forward_kinematics(sol, dh_table)
        pos_error = np.linalg.norm(T_fk[:3, 3] - Td[:3, 3])
        rot_error = np.linalg.norm(T_fk[:3, :3] - Td[:3, :3])

        sol_deg = np.round(np.degrees(sol), 2)
        config = f"Config {idx + 1}"
        print(f"{config} (degrees): {sol_deg.tolist()}")
        print(f"{config} - Pos Error: {pos_error:.6f}, Rot Error: {rot_error:.6f}")

        if pos_error < tol and rot_error < tol:
            valid_count += 1
            valid_configs.append(idx + 1)

    print(f"\nNumber of valid IK configurations with zero error: {valid_count} out of {len(solutions)}\n")
    if valid_configs:
        print(f"Valid configurations: {', '.join('Config ' + str(c) for c in valid_configs)}\n")
    else:
        print("No valid configurations found.\n")

def generate_random_pose(x_range=(-1.2, 1.2),
                         y_range=(-1.0, 1.0),
                         z_range=(0.2, 2.0)):
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    z = np.random.uniform(*z_range)

    roll  = np.random.uniform(-np.pi, np.pi)
    pitch = np.random.uniform(-np.pi, np.pi)
    yaw   = np.random.uniform(-np.pi, np.pi)

    return [x, y, z, roll, pitch, yaw]


def gantry_to_arm_base(q_gantry, robot):
    # Gantry translation along X-axis
    T_gantry = np.eye(4)
    T_gantry[0, 3] = q_gantry  # prismatic joint displacement along X
    T_flange_offset = np.eye(4)
    for j in robot.joints:
        T_flange_offset = T_flange_offset @ j.origin

    # Total transform: gantry position + flange offset
    T_arm_base = T_gantry @ T_flange_offset
    return T_arm_base


def compute_jacobian(q, dh_table, eps=1e-6):
    n = len(q)
    J = np.zeros((6, n))
    T0 = forward_kinematics(q, dh_table)

    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        T_eps = forward_kinematics(q + dq, dh_table)

        # Position difference
        dp = (T_eps[:3, 3] - T0[:3, 3]) / eps

        # Rotation difference (as axis-angle derivative)
        dR = T0[:3, :3].T @ T_eps[:3, :3]
        dtheta = Rotation.from_matrix(dR).as_rotvec() / eps

        J[:3, i] = dp
        J[3:, i] = dtheta

    return J

def numerical_ik(target_T, dh_table, q_init=None, max_iters=100, tol=1e-4, damping=1e-3):
    n_joints = len(dh_table) - 1  # exclude flange link
    if q_init is None:
        q = np.zeros(n_joints)
    else:
        q = np.array(q_init, dtype=float)

    for i in range(max_iters):
        T_fk = forward_kinematics(q, dh_table)

        # Compute 6D error
        pos_err = target_T[:3, 3] - T_fk[:3, 3]
        R_err = T_fk[:3, :3].T @ target_T[:3, :3]
        rot_err = Rotation.from_matrix(R_err).as_rotvec()

        error = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(error) < tol:
            return q  # Converged

        J = compute_jacobian(q, dh_table)

        # Damped Least Squares solution
        JTJ = J.T @ J + damping ** 2 * np.eye(n_joints)
        dq = np.linalg.solve(JTJ, J.T @ error)

        q += dq

    print("[Numerical IK] Failed to converge.")
    return None


if __name__ == "__main__":

    use_custom_pose = input("Do you want to input your own pose? (y/n): ").strip().lower()
    if use_custom_pose == 'y':
        try:
            user_input = input(
                "Enter 6 values separated by space (x y z in meters, roll pitch yaw in degrees): "
            )
            values = list(map(float, user_input.split()))
            if len(values) != 6:
                raise ValueError("You must enter exactly 6 numeric values.")

            # Convert Euler angles from degrees to radians
            pose = [
                values[0], values[1], values[2],
                np.radians(values[3]), np.radians(values[4]), np.radians(values[5])
            ]
        except Exception as e:
            print(f"Invalid input: {e}. Using a random pose instead.")
            pose = generate_random_pose()
    else:
        pose = generate_random_pose()

    print("Random Pose:", np.round(pose, 3))

    urdf_arm = "robot-urdfs/abb_irb6700_150_320/abb_irb6700_150_320.urdf"
    urdf_gantry = "robot-urdfs/linear_axis/linear_axis.urdf"
    arm = URDF.load("robot-urdfs/abb_irb6700_150_320/abb_irb6700_150_320.urdf")
    gantry = URDF.load("robot-urdfs/linear_axis/linear_axis.urdf")
    dh_table = extract_modified_dh(arm)

    # Align the base with the target pose
    q_gantry = pose[0]

    # Compute arm base transform given gantry position
    T_arm_base = gantry_to_arm_base(q_gantry, gantry)

    # Desired EE pose in global frame
    Td_global = pose_to_transform(pose)

    # Convert desired EE pose to arm base frame for IK
    Td_arm = np.linalg.inv(T_arm_base) @ Td_global

    solutions = inverse_kinematics(Td_arm, dh_table)

    print(f"Gantry position (q_gantry): {q_gantry:.3f} m")

    check_ik_solutions(solutions, Td_arm, dh_table)

    q_sol = numerical_ik(Td_arm, dh_table)

    if q_sol is not None:
        q_sol = np.array([wrap_to_pi(q) for q in q_sol])
        print("Numerical IK solution (degrees):", np.round(np.degrees(q_sol), 2))
        T_check = forward_kinematics(q_sol, dh_table)
        pos_error = np.linalg.norm(T_check[:3, 3] - Td_arm[:3, 3])
        print(f"Position error: {pos_error:.6f} m")
        joint_dict = {j.name: float(q) for j, q in zip(arm.joints, q_sol)}

# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz


def _in_corridor_vec(x, y, z):
    """Vectorized corridor check. Returns bool array."""
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return r1 | r2 | r3


def _densify_path(full_path):
    """Densify path so each step moves at most 0.05 Euclidean and changes each angle by at most 2 degrees."""
    densified = [full_path[0]]
    for i in range(1, len(full_path)):
        p0 = full_path[i - 1]
        p1 = full_path[i]
        pos_dist = np.linalg.norm(p1[:3] - p0[:3])
        ang_diffs = np.abs(p1[3:] - p0[3:])
        n_steps = max(1, int(np.ceil(pos_dist / 0.05)), int(np.ceil(np.max(ang_diffs) / 2.0)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = p0 + t * (p1 - p0)
            densified.append(interp)
    return np.array(densified)


def _build_grid():
    """Build the grid of points once."""
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return points


def _rotation_arc_bounding_box_in_corridor(points_alive, R_start, R_end, center, axis, angle_change):
    """
    Check if the axis-aligned bounding box of the rotation arc lies in the corridor.
    For a single-axis rotation, the bounding box can be computed analytically.
    """
    if len(points_alive) == 0:
        return True
    
    # Sample points along the rotation arc
    n_samples = max(2, int(np.ceil(abs(angle_change) / 5.0)))
    all_points = []
    for i in range(n_samples + 1):
        t = i / n_samples
        if axis == 0:  # yaw
            R_t = _rotation_matrix(R_start[3] + t * angle_change, R_start[4], R_start[5])
        elif axis == 1:  # pitch
            R_t = _rotation_matrix(R_start[3], R_start[4] + t * angle_change, R_start[5])
        else:  # roll
            R_t = _rotation_matrix(R_start[3], R_start[4], R_start[5] + t * angle_change)
        placed = points_alive @ R_t.T + center
        all_points.append(placed)
    
    all_points = np.vstack(all_points)
    return np.all(_in_corridor_vec(all_points[:, 0], all_points[:, 1], all_points[:, 2]))


def _evaluate_path_fast(path, points=None):
    """
    Fast vectorized evaluation of the path.
    Returns the surviving volume (count of 4x4x4 blocks * 0.05^3).
    """
    if points is None:
        points = _build_grid()
    
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    densified = _densify_path(full_path)
    
    N = len(points)
    alive = np.ones(N, dtype=bool)
    
    # Check at first pose
    p0 = densified[0]
    R0 = _rotation_matrix(p0[3], p0[4], p0[5])
    placed0 = points @ R0.T + p0[:3]
    alive &= _in_corridor_vec(placed0[:, 0], placed0[:, 1], placed0[:, 2])
    
    # Process each step
    for step in range(1, len(densified)):
        if not np.any(alive):
            break
        prev = densified[step - 1]
        curr = densified[step]
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dz = curr[2] - prev[2]
        dyaw = curr[3] - prev[3]
        dpitch = curr[4] - prev[4]
        droll = curr[5] - prev[5]
        
        # x move
        if abs(dx) > 1e-12:
            p_x = prev.copy()
            p_x[0] = curr[0]
            R_x = _rotation_matrix(p_x[3], p_x[4], p_x[5])
            placed_x = points[alive] @ R_x.T + p_x[:3]
            new_alive = _in_corridor_vec(placed_x[:, 0], placed_x[:, 1], placed_x[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # y move
        if abs(dy) > 1e-12:
            p_y = prev.copy()
            p_y[0] = curr[0]
            p_y[1] = curr[1]
            R_y = _rotation_matrix(p_y[3], p_y[4], p_y[5])
            placed_y = points[alive] @ R_y.T + p_y[:3]
            new_alive = _in_corridor_vec(placed_y[:, 0], placed_y[:, 1], placed_y[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # z move
        if abs(dz) > 1e-12:
            p_z = prev.copy()
            p_z[0] = curr[0]
            p_z[1] = curr[1]
            p_z[2] = curr[2]
            R_z = _rotation_matrix(p_z[3], p_z[4], p_z[5])
            placed_z = points[alive] @ R_z.T + p_z[:3]
            new_alive = _in_corridor_vec(placed_z[:, 0], placed_z[:, 1], placed_z[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
        
        # Rotation - single axis with bounding box check
        if abs(dyaw) > 1e-12 and abs(dpitch) < 1e-12 and abs(droll) < 1e-12:
            p_rot = prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[3] = curr[3]
            # Check bounding box of rotation arc
            if not _rotation_arc_bounding_box_in_corridor(points[alive], prev, curr, p_rot[:3], 0, dyaw):
                alive[:] = False
                break
            n_samples = max(2, int(np.ceil(abs(dyaw) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                yaw_s = prev[3] + t * dyaw
                R_s = _rotation_matrix(yaw_s, p_rot[4], p_rot[5])
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(dpitch) > 1e-12 and abs(dyaw) < 1e-12 and abs(droll) < 1e-12:
            p_rot = prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[4] = curr[4]
            if not _rotation_arc_bounding_box_in_corridor(points[alive], prev, curr, p_rot[:3], 1, dpitch):
                alive[:] = False
                break
            n_samples = max(2, int(np.ceil(abs(dpitch) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                pitch_s = prev[4] + t * dpitch
                R_s = _rotation_matrix(p_rot[3], pitch_s, p_rot[5])
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(droll) > 1e-12 and abs(dyaw) < 1e-12 and abs(dpitch) < 1e-12:
            p_rot = prev.copy()
            p_rot[0] = curr[0]
            p_rot[1] = curr[1]
            p_rot[2] = curr[2]
            p_rot[5] = curr[5]
            if not _rotation_arc_bounding_box_in_corridor(points[alive], prev, curr, p_rot[:3], 2, droll):
                alive[:] = False
                break
            n_samples = max(2, int(np.ceil(abs(droll) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                roll_s = prev[5] + t * droll
                R_s = _rotation_matrix(p_rot[3], p_rot[4], roll_s)
                placed_s = points[alive] @ R_s.T + p_rot[:3]
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        else:
            # Multiple axes or complex - check final pose
            R_rot = _rotation_matrix(curr[3], curr[4], curr[5])
            placed_rot = points[alive] @ R_rot.T + curr[:3]
            new_alive = _in_corridor_vec(placed_rot[:, 0], placed_rot[:, 1], placed_rot[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
    
    # Count 4x4x4 blocks
    alive_3d = alive.reshape(81, 21, 21)
    blocks = sliding_window_view(alive_3d, (4, 4, 4))
    all_survive = np.all(blocks, axis=(3, 4, 5))
    count = np.sum(all_survive)
    return count * (0.05 ** 3)


def _generate_medial_axis_path():
    """
    Generate a path that follows the medial axis of the corridor.
    The corridor is L-shaped with a vertical extension.
    """
    poses = []
    
    # Start in region 1 (x<=0, 0<=y<=1, 0<=z<=1)
    # Move along x from -3 to 0, keeping y=0.5, z=0.5
    for x in np.linspace(-3, 0, 13):
        poses.append([x, 0.5, 0.5, 0, 0, 0])
    
    # Transition to region 2: rotate yaw from 0 to 90 degrees
    # While moving in y from 0.5 to 2.5
    for i in range(21):
        t = i / 20
        yaw = 90 * t
        y = 0.5 + 2.0 * t
        x = 0.5
        z = 0.5
        poses.append([x, y, z, yaw, 0, 0])
    
    # Move along y from 2.5 to 5 in region 2
    for y in np.linspace(2.5, 5, 13):
        poses.append([0.5, y, 0.5, 90, 0, 0])
    
    # Transition to region 3: rotate pitch from 0 to 90 degrees
    # While moving in z from 0.5 to 2.5
    for i in range(21):
        t = i / 20
        pitch = 90 * t
        z = 0.5 + 2.0 * t
        x = 0.5
        y = 5.5
        poses.append([x, y, z, 90, pitch, 0])
    
    # Move along z from 2.5 to 7 in region 3
    for z in np.linspace(2.5, 7, 13):
        poses.append([0.5, 5.5, z, 90, 90, 0])
    
    # Final rotation to match end pose (90, 90, 90)
    for i in range(11):
        t = i / 10
        roll = 90 * t
        poses.append([0.5, 5.5, 7, 90, 90, roll])
    
    return np.array(poses)


def _seed_path():
    """Return the seed path as a numpy array. Uses medial axis path."""
    return _generate_medial_axis_path()


def _optimize_pose_positions(path, points, time_limit, rng):
    """
    Optimize the position components (x,y,z) of each pose to maximize survival.
    We focus on keeping the rotated grid points centered in the corridor.
    """
    best_path = path.copy()
    best_score = _evaluate_path_fast(best_path, points)
    
    # For each pose, try to adjust position to maximize local survival
    # Strategy: for each pose, the ideal position keeps the centroid of 
    # rotated points at the corridor center
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    centroid_orig = np.array([np.mean(X), np.mean(Y), np.mean(Z)])
    
    for iter_num in range(200):
        if time.time() > time_limit:
            break
        
        # Pick a random pose to optimize
        idx = rng.integers(0, len(best_path))
        pose = best_path[idx].copy()
        
        # Compute rotated centroid
        R = _rotation_matrix(pose[3], pose[4], pose[5])
        rotated_centroid = R @ centroid_orig
        
        # Ideal position: keep rotated centroid at corridor center
        # In region 2 (main corridor), center is roughly (0.5, y_center, 0.5)
        # In region 1, center is (-x, 0.5, 0.5)
        # In region 3, center is (0.5, 5.5, z_center)
        
        # Try small perturbations around current position
        best_local_score = best_score
        best_local_pose = pose.copy()
        
        for trial in range(10):
            trial_pose = pose.copy()
            trial_pose[0] += rng.normal(0, 0.02)
            trial_pose[1] += rng.normal(0, 0.02)
            trial_pose[2] += rng.normal(0, 0.02)
            
            test_path = best_path.copy()
            test_path[idx] = trial_pose
            trial_score = _evaluate_path_fast(test_path, points)
            
            if trial_score > best_local_score:
                best_local_score = trial_score
                best_local_pose = trial_pose
        
        if best_local_score > best_score:
            best_path[idx] = best_local_pose
            best_score = best_local_score
    
    return best_path, best_score


def _optimize_pose_angles(path, points, time_limit, rng):
    """
    Optimize the angle components (yaw, pitch, roll) of each pose.
    Focus on keeping rotations minimal to preserve more points in corridor.
    """
    best_path = path.copy()
    best_score = _evaluate_path_fast(best_path, points)
    
    for iter_num in range(200):
        if time.time() > time_limit:
            break
        
        idx = rng.integers(0, len(best_path))
        pose = best_path[idx].copy()
        
        best_local_score = best_score
        best_local_pose = pose.copy()
        
        for trial in range(10):
            trial_pose = pose.copy()
            # Small angle perturbations in degrees
            trial_pose[3] += rng.normal(0, 0.5)
            trial_pose[4] += rng.normal(0, 0.5)
            trial_pose[5] += rng.normal(0, 0.5)
            
            test_path = best_path.copy()
            test_path[idx] = trial_pose
            trial_score = _evaluate_path_fast(test_path, points)
            
            if trial_score > best_local_score:
                best_local_score = trial_score
                best_local_pose = trial_pose
        
        if best_local_score > best_score:
            best_path[idx] = best_local_pose
            best_score = best_local_score
    
    return best_path, best_score


def _optimize_add_remove_poses(path, points, time_limit, rng):
    """
    Try adding or removing poses to better control the trajectory.
    """
    best_path = path.copy()
    best_score = _evaluate_path_fast(best_path, points)
    
    for iter_num in range(50):
        if time.time() > time_limit:
            break
        
        action = rng.integers(0, 2)  # 0: add, 1: remove
        
        if action == 0 and len(best_path) < 150:
            # Add a pose between two existing ones
            if len(best_path) >= 2:
                idx = rng.integers(0, len(best_path) - 1)
                p0 = best_path[idx]
                p1 = best_path[idx + 1]
                t = rng.uniform(0.2, 0.8)
                new_pose = p0 + t * (p1 - p0)
                new_path = np.vstack([best_path[:idx+1], new_pose, best_path[idx+1:]])
                new_score = _evaluate_path_fast(new_path, points)
                if new_score > best_score:
                    best_path = new_path
                    best_score = new_score
        
        elif action == 1 and len(best_path) > 10:
            # Remove a pose
            idx = rng.integers(1, len(best_path) - 1)  # Don't remove first or last
            new_path = np.vstack([best_path[:idx], best_path[idx+1:]])
            new_score = _evaluate_path_fast(new_path, points)
            if new_score > best_score:
                best_path = new_path
                best_score = new_score
    
    return best_path, best_score


def moving_sofa_3d_path() -> np.ndarray:
    """
    The 3D moving sofa problem: model-and-optimize approach.
    Parameterize the path and optimize directly against the task objective.
    """
    t_start = time.time()
    deadline = t_start + 600.0  # 600s limit, leave 30s margin
    
    points = _build_grid()
    rng = np.random.default_rng(42)
    
    # Start with medial axis path
    best_path = _generate_medial_axis_path()
    best_score = _evaluate_path_fast(best_path, points)
    
    # Phase 1: Optimize positions of existing poses
    time_limit_1 = t_start + 150
    best_path, best_score = _optimize_pose_positions(best_path, points, time_limit_1, rng)
    
    # Phase 2: Optimize angles
    time_limit_2 = t_start + 300
    best_path, best_score = _optimize_pose_angles(best_path, points, time_limit_2, rng)
    
    # Phase 3: Add/remove poses
    time_limit_3 = t_start + 450
    best_path, best_score = _optimize_add_remove_poses(best_path, points, time_limit_3, rng)
    
    # Phase 4: Final refinement - joint position+angle optimization
    time_limit_4 = deadline - 30
    best_path_final = best_path.copy()
    best_score_final = best_score
    
    for iter_num in range(500):
        if time.time() > time_limit_4:
            break
        
        idx = rng.integers(0, len(best_path_final))
        trial_pose = best_path_final[idx].copy()
        
        # Joint perturbation
        trial_pose[0] += rng.normal(0, 0.01)
        trial_pose[1] += rng.normal(0, 0.01)
        trial_pose[2] += rng.normal(0, 0.01)
        trial_pose[3] += rng.normal(0, 0.3)
        trial_pose[4] += rng.normal(0, 0.3)
        trial_pose[5] += rng.normal(0, 0.3)
        
        test_path = best_path_final.copy()
        test_path[idx] = trial_pose
        trial_score = _evaluate_path_fast(test_path, points)
        
        if trial_score > best_score_final:
            best_path_final = test_path
            best_score_final = trial_score
    
    return best_path_final


# EVOLVE-BLOCK-END

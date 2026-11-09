# EVOLVE-BLOCK-START
import numpy as np
import time


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
        max_ang = np.max(ang_diffs)
        n_steps = max(1, int(np.ceil(pos_dist / 0.05)), int(np.ceil(max_ang / 2.0)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = p0 + t * (p1 - p0)
            densified.append(interp)
    return np.array(densified)


def _evaluate_path_fast(path):
    """
    Fast vectorized evaluation of the path.
    Returns the surviving volume (count of 4x4x4 blocks * 0.05^3).
    """
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    densified = _densify_path(full_path)
    
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N = len(points)
    
    alive = np.ones(N, dtype=bool)
    
    p0 = densified[0]
    R0 = _rotation_matrix(p0[3], p0[4], p0[5])
    placed0 = points @ R0.T + p0[:3]
    alive &= _in_corridor_vec(placed0[:, 0], placed0[:, 1], placed0[:, 2])
    
    # Track the current position after translation moves
    cur_pos = densified[0][:3].copy()
    
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
        
        # Translation moves: x, y, z sequentially
        if abs(dx) > 1e-12:
            p_x = prev.copy()
            p_x[0] = curr[0]
            R_x = _rotation_matrix(p_x[3], p_x[4], p_x[5])
            placed_x = points[alive] @ R_x.T + p_x[:3]
            new_alive = _in_corridor_vec(placed_x[:, 0], placed_x[:, 1], placed_x[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
            cur_pos[0] = curr[0]
        
        if abs(dy) > 1e-12:
            p_y = prev.copy()
            p_y[0] = curr[0]
            p_y[1] = curr[1]
            R_y = _rotation_matrix(p_y[3], p_y[4], p_y[5])
            placed_y = points[alive] @ R_y.T + p_y[:3]
            new_alive = _in_corridor_vec(placed_y[:, 0], placed_y[:, 1], placed_y[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
            cur_pos[1] = curr[1]
        
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
            cur_pos[2] = curr[2]
        
        # Rotation moves: determine which axis
        if abs(dyaw) > 1e-12 and abs(dpitch) < 1e-12 and abs(droll) < 1e-12:
            p_rot = cur_pos.copy()
            n_samples = max(2, int(np.ceil(abs(dyaw) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                yaw_s = prev[3] + t * dyaw
                R_s = _rotation_matrix(yaw_s, curr[4], curr[5])
                placed_s = points[alive] @ R_s.T + p_rot
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(dpitch) > 1e-12 and abs(dyaw) < 1e-12 and abs(droll) < 1e-12:
            p_rot = cur_pos.copy()
            n_samples = max(2, int(np.ceil(abs(dpitch) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                pitch_s = prev[4] + t * dpitch
                R_s = _rotation_matrix(curr[3], pitch_s, curr[5])
                placed_s = points[alive] @ R_s.T + p_rot
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        elif abs(droll) > 1e-12 and abs(dyaw) < 1e-12 and abs(dpitch) < 1e-12:
            p_rot = cur_pos.copy()
            n_samples = max(2, int(np.ceil(abs(droll) / 2.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                roll_s = prev[5] + t * droll
                R_s = _rotation_matrix(curr[3], curr[4], roll_s)
                placed_s = points[alive] @ R_s.T + p_rot
                new_alive = _in_corridor_vec(placed_s[:, 0], placed_s[:, 1], placed_s[:, 2])
                alive_idx = np.where(alive)[0]
                alive[alive_idx[~new_alive]] = False
        else:
            # Multiple angle changes or no change - just check final pose
            R_rot = _rotation_matrix(curr[3], curr[4], curr[5])
            placed_rot = points[alive] @ R_rot.T + cur_pos
            new_alive = _in_corridor_vec(placed_rot[:, 0], placed_rot[:, 1], placed_rot[:, 2])
            alive_idx = np.where(alive)[0]
            alive[alive_idx[~new_alive]] = False
    
    alive_3d = alive.reshape(81, 21, 21)
    from numpy.lib.stride_tricks import sliding_window_view
    blocks = sliding_window_view(alive_3d, (4, 4, 4))
    all_survive = np.all(blocks, axis=(3, 4, 5))
    count = np.sum(all_survive)
    return count * (0.05 ** 3)


def _build_medial_axis_path():
    """
    Build a path that follows the medial axis of the corridor, snapping to
    geometric features. This ensures the sofa stays as centered as possible
    in each corridor segment, maximizing the surviving volume.
    
    The corridor has three regions:
    - Region 1: x<=0, 0<=y<=1, 0<=z<=1 (entrance, oriented along x)
    - Region 2: 0<=x<=1, 0<=y<=5, 0<=z<=1 (middle, oriented along y)
    - Region 3: 0<=x<=1, 5<=y<=6, z>=0 (exit, oriented along z)
    
    The medial axis for each region:
    - Region 1: y=0.5, z=0.5, x varies from -3 to 0
    - Region 2: x=0.5, z=0.5, y varies from 0 to 5
    - Region 3: x=0.5, y=5.5, z varies from 0 to 7
    
    At the corners, we rotate in place at the center of the 1x1 cross-section.
    """
    path = []
    
    # Phase 1: Enter region 1, moving in +x direction
    # Sofa is oriented with yaw=0, pitch=0, roll=0
    # y and z should be centered at 0.5
    for x in np.linspace(-3.0, -0.5, 30):
        path.append([x, 0.5, 0.5, 0.0, 0.0, 0.0])
    
    # Move to the first corner center: x=0.5, y=0.5, z=0.5
    path.append([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    
    # Phase 2: First turn - rotate yaw from 0 to 90 at the corner
    # Position at corner: x=0.5, y=0.5, z=0.5 (center of the 1x1 corner)
    for yaw in np.linspace(0, 90, 50):
        path.append([0.5, 0.5, 0.5, yaw, 0.0, 0.0])
    
    # Phase 3: Travel along y in region 2
    # Now facing +y direction (yaw=90)
    # x and z should be centered at 0.5
    for y in np.linspace(0.5, 5.0, 50):
        path.append([0.5, y, 0.5, 90.0, 0.0, 0.0])
    
    # Move to the second corner center: x=0.5, y=5.5, z=0.5
    path.append([0.5, 5.5, 0.5, 90.0, 0.0, 0.0])
    
    # Phase 4: Second turn - rotate to face +z
    # At the second corner: x=0.5, y=5.5, z=0.5
    # Need to rotate pitch from 0 to 90, and roll from 0 to 90
    # When yaw=90, pitch=90, roll=90: the sofa faces +z direction
    for pitch in np.linspace(0, 90, 50):
        path.append([0.5, 5.5, 0.5, 90.0, pitch, 0.0])
    
    for roll in np.linspace(0, 90, 50):
        path.append([0.5, 5.5, 0.5, 90.0, 90.0, roll])
    
    # Phase 5: Travel in +z direction in region 3
    for z in np.linspace(0.5, 6.5, 50):
        path.append([0.5, 5.5, z, 90.0, 90.0, 90.0])
    
    return np.array(path, dtype=np.float64)


def _build_smoothed_path(path, n_refine=5):
    """
    Refine a path by smoothing it to reduce sharp corners and ensure
    the rotation arcs stay within the corridor.
    """
    # For now, just return the path as-is
    # Future: could apply Gaussian smoothing or other techniques
    return path


def _local_search_refine(path, rng, max_iterations=2000, deadline=None):
    """
    Local search to refine the path by perturbing individual poses.
    """
    best_path = path.copy()
    best_score = _evaluate_path_fast(best_path)
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    t_start = time.time()
    if deadline is None:
        deadline = t_start + 580.0
    
    for i in range(max_iterations):
        if time.time() > deadline - 20:
            break
        if no_improve_count >= max_no_improve:
            # Reset step size and try again
            step_size = max(0.01, step_size * 0.7)
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)
        
        # Randomly perturb a few poses
        n_perturb = int(rng.integers(1, 4))
        n_poses = len(best_path)
        if n_poses <= n_perturb:
            indices = np.arange(n_poses)
        else:
            indices = rng.choice(n_poses, size=n_perturb, replace=False)
        
        new_path = best_path.copy()
        for idx in indices:
            new_path[idx, 0] += rng.normal(0, step_size)
            new_path[idx, 1] += rng.normal(0, step_size)
            new_path[idx, 2] += rng.normal(0, step_size)
            new_path[idx, 3] += rng.normal(0, step_size * 2)
            new_path[idx, 4] += rng.normal(0, step_size * 2)
            new_path[idx, 5] += rng.normal(0, step_size * 2)
        
        new_score = _evaluate_path_fast(new_path)
        
        if new_score > best_score:
            best_path = new_path
            best_score = new_score
            no_improve_count = 0
            step_size *= 0.95
        else:
            no_improve_count += 1
            if no_improve_count % 20 == 0:
                step_size *= 1.1
    
    return best_path, best_score


def moving_sofa_3d_path() -> np.ndarray:
    """
    The 3D moving sofa problem: find the largest-volume rigid 3D shape that can
    travel through an L-shaped 3D corridor of unit cross-section that has two
    90-degree turns.
    
    This implementation builds a path following the medial axis of the corridor,
    then refines it with local search to maximize the surviving volume.
    
    Returns:
        path: np.ndarray of shape (k, 6), each row a pose
            (x, y, z, yaw_deg, pitch_deg, roll_deg).
    """
    t_start = time.time()
    deadline = t_start + 580.0  # 630s limit, leave 50s margin
    
    # Start with the medial axis path
    medial_path = _build_medial_axis_path()
    best_path = medial_path.copy()
    best_score = _evaluate_path_fast(best_path)
    
    # Try a few different initial paths and pick the best
    # Path variant 1: slightly offset from center
    path_var1 = medial_path.copy()
    # Shift the y positions in region 2 slightly
    for i in range(len(path_var1)):
        if 0.5 <= path_var1[i, 1] <= 5.0 and path_var1[i, 3] == 90.0:
            path_var1[i, 1] += 0.05
    
    score_var1 = _evaluate_path_fast(path_var1)
    if score_var1 > best_score:
        best_path = path_var1
        best_score = score_var1
    
    # Local search to refine
    rng = np.random.default_rng(42)
    best_path, best_score = _local_search_refine(best_path, rng, max_iterations=3000, deadline=deadline)
    
    return best_path


# EVOLVE-BLOCK-END

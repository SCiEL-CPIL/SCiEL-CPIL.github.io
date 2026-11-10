import numpy as np
import time

def _hex_vertices_batch(positions, angles):
    """Compute vertices for all hexagons. positions: (N,2), angles: (N,) in degrees."""
    a = np.radians(angles)
    base_angles = np.radians(60 * np.arange(6))
    all_angles = a[:, None] + base_angles[None, :]
    cos_a = np.cos(all_angles)
    sin_a = np.sin(all_angles)
    x = positions[:, 0:1] + 1.0 * cos_a
    y = positions[:, 1:2] + 1.0 * sin_a
    return np.stack([x, y], axis=-1)  # shape (N, 6, 2)

def _get_edge_normals(verts):
    """Get 6 edge normals for a single hexagon. verts: (6, 2)."""
    edges = np.roll(verts, -1, axis=0) - verts
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return normals / norms

def _hex_separated_fast(v1, v2, tol=1e-6):
    """Check if two hexagons are separated. v1, v2: (6, 2)."""
    n1 = _get_edge_normals(v1)
    n2 = _get_edge_normals(v2)
    p1_1 = v1 @ n1.T
    p2_1 = v2 @ n1.T
    for i in range(6):
        if np.max(p1_1[:, i]) < np.min(p2_1[:, i]) - tol or np.max(p2_1[:, i]) < np.min(p1_1[:, i]) - tol:
            return True
    p1_2 = v1 @ n2.T
    p2_2 = v2 @ n2.T
    for i in range(6):
        if np.max(p1_2[:, i]) < np.min(p2_2[:, i]) - tol or np.max(p2_2[:, i]) < np.min(p1_2[:, i]) - tol:
            return True
    return False

def _is_valid_packing(positions, angles):
    """Check if all pairs of hexagons are separated."""
    verts = _hex_vertices_batch(positions, angles)
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hex_separated_fast(verts[i], verts[j]):
                return False
    return True

def _compute_outer_side(positions, angles, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (centered at origin)
    with the given angle that contains all inner hexagon vertices.
    """
    verts = _hex_vertices_batch(positions, angles)
    all_pts = verts.reshape(-1, 2)
    
    a = np.radians(outer_angle)
    max_abs_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_pts @ n
        max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
    
    return 2.0 * max_abs_proj / np.sqrt(3)

def _build_11_configs():
    """
    Build multiple valid 11-hexagon configurations.
    """
    sqrt3 = np.sqrt(3)
    configs = []
    
    # Config 1: Hex-7 cluster + 4 at corners (classic)
    pos1 = np.zeros((11, 2))
    pos1[0] = [0, 0]
    pos1[1] = [2, 0]
    pos1[2] = [1, sqrt3]
    pos1[3] = [-1, sqrt3]
    pos1[4] = [-2, 0]
    pos1[5] = [-1, -sqrt3]
    pos1[6] = [1, -sqrt3]
    pos1[7] = [3, sqrt3]
    pos1[8] = [-3, sqrt3]
    pos1[9] = [3, -sqrt3]
    pos1[10] = [-3, -sqrt3]
    configs.append((pos1, np.zeros(11)))
    
    # Config 2: Compact diamond extension
    pos2 = np.zeros((11, 2))
    pos2[0] = [0, 0]
    pos2[1] = [2, 0]
    pos2[2] = [1, sqrt3]
    pos2[3] = [-1, sqrt3]
    pos2[4] = [-2, 0]
    pos2[5] = [-1, -sqrt3]
    pos2[6] = [1, -sqrt3]
    pos2[7] = [0, 2*sqrt3]
    pos2[8] = [0, -2*sqrt3]
    pos2[9] = [3, sqrt3/2]
    pos2[10] = [-3, -sqrt3/2]
    configs.append((pos2, np.zeros(11)))
    
    # Config 3: Linear chain with branches
    pos3 = np.zeros((11, 2))
    pos3[0] = [0, 0]
    pos3[1] = [2, 0]
    pos3[2] = [4, 0]
    pos3[3] = [6, 0]
    pos3[4] = [1, sqrt3]
    pos3[5] = [-1, sqrt3]
    pos3[6] = [-2, 0]
    pos3[7] = [3, sqrt3]
    pos3[8] = [5, sqrt3]
    pos3[9] = [1, -sqrt3]
    pos3[10] = [-1, -sqrt3]
    configs.append((pos3, np.zeros(11)))
    
    # Config 4: 2x3 grid + extras
    pos4 = np.zeros((11, 2))
    pos4[0] = [0, 0]
    pos4[1] = [2, 0]
    pos4[2] = [4, 0]
    pos4[3] = [1, sqrt3]
    pos4[4] = [3, sqrt3]
    pos4[5] = [5, sqrt3]
    pos4[6] = [1, -sqrt3]
    pos4[7] = [3, -sqrt3]
    pos4[8] = [5, -sqrt3]
    pos4[9] = [2, 2*sqrt3]
    pos4[10] = [4, 2*sqrt3]
    configs.append((pos4, np.zeros(11)))
    
    # Config 5: Asymmetric tight
    pos5 = np.zeros((11, 2))
    pos5[0] = [0, 0]
    pos5[1] = [2, 0]
    pos5[2] = [1, sqrt3]
    pos5[3] = [-1, sqrt3]
    pos5[4] = [-2, 0]
    pos5[5] = [-1, -sqrt3]
    pos5[6] = [1, -sqrt3]
    pos5[7] = [3.5, sqrt3 * 0.5]
    pos5[8] = [-3.5, sqrt3 * 0.5]
    pos5[9] = [2, sqrt3 * 2]
    pos5[10] = [-2, sqrt3 * 2]
    configs.append((pos5, np.zeros(11)))
    
    # Config 6: Staggered ring variant
    pos6 = np.zeros((11, 2))
    pos6[0] = [0, 0]
    pos6[1] = [2, 0]
    pos6[2] = [1, sqrt3]
    pos6[3] = [-1, sqrt3]
    pos6[4] = [-2, 0]
    pos6[5] = [-1, -sqrt3]
    pos6[6] = [1, -sqrt3]
    pos6[7] = [3, 0]
    pos6[8] = [-3, 0]
    pos6[9] = [1.5, sqrt3 * 2.5]
    pos6[10] = [-1.5, -sqrt3 * 2.5]
    configs.append((pos6, np.zeros(11)))
    
    # Config 7: Compact 3-4-4 layout
    pos7 = np.zeros((11, 2))
    pos7[0] = [0, 0]
    pos7[1] = [2, 0]
    pos7[2] = [-2, 0]
    pos7[3] = [1, sqrt3]
    pos7[4] = [-1, sqrt3]
    pos7[5] = [3, sqrt3]
    pos7[6] = [-3, sqrt3]
    pos7[7] = [1, -sqrt3]
    pos7[8] = [-1, -sqrt3]
    pos7[9] = [3, -sqrt3]
    pos7[10] = [-3, -sqrt3]
    configs.append((pos7, np.zeros(11)))
    
    # Config 8: T-shaped
    pos8 = np.zeros((11, 2))
    pos8[0] = [0, 0]
    pos8[1] = [2, 0]
    pos8[2] = [4, 0]
    pos8[3] = [6, 0]
    pos8[4] = [-2, 0]
    pos8[5] = [-4, 0]
    pos8[6] = [2, sqrt3]
    pos8[7] = [2, -sqrt3]
    pos8[8] = [4, sqrt3]
    pos8[9] = [4, -sqrt3]
    pos8[10] = [0, sqrt3]
    configs.append((pos8, np.zeros(11)))
    
    return configs

def _local_search_random(pos_init, angles_init, deadline, rng, max_iters=5000):
    """
    Random perturbation local search with adaptive step size.
    """
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    best_side = _compute_outer_side(best_pos, best_angles, 0.0)
    best_outer_angle = 0.0
    
    # Find best outer angle for initial config
    for outer_a in np.linspace(0, 60, 30):
        side = _compute_outer_side(best_pos, best_angles, outer_a)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_a
    
    step_size = 0.3
    consecutive_fails = 0
    max_consecutive_fails = 30
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        new_pos = best_pos.copy()
        new_angles = best_angles.copy()
        
        # Randomly perturb 1-3 hexagons
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(11, size=n_perturb, replace=False)
        for idx in indices:
            new_pos[idx] += rng.normal(0, step_size, size=2)
            new_angles[idx] += rng.normal(0, step_size * 5)
        
        if _is_valid_packing(new_pos, new_angles):
            # Find best outer angle for this config (focused search)
            best_local_side = best_side
            best_local_outer = best_outer_angle
            for outer_a in np.linspace(max(0, best_outer_angle - 15), min(60, best_outer_angle + 15), 20):
                side = _compute_outer_side(new_pos, new_angles, outer_a)
                if side < best_local_side:
                    best_local_side = side
                    best_local_outer = outer_a
            
            if best_local_side < best_side - 1e-9:
                best_side = best_local_side
                best_pos = new_pos
                best_angles = new_angles
                best_outer_angle = best_local_outer
                consecutive_fails = 0
                step_size = max(0.02, step_size * 0.95)
            else:
                consecutive_fails += 1
                if consecutive_fails > max_consecutive_fails:
                    step_size = min(0.5, step_size * 1.08)
                    consecutive_fails = 0
        else:
            consecutive_fails += 1
    
    return best_side, best_pos, best_angles, best_outer_angle

def _refine_fine(pos_init, angles_init, outer_angle_init, deadline, rng, max_iters=3000):
    """
    High-precision refinement with small perturbations.
    """
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    best_outer_angle = outer_angle_init
    best_side = _compute_outer_side(best_pos, best_angles, best_outer_angle)
    
    step_size = 0.08
    consecutive_fails = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        idx = rng.integers(0, 11)
        new_pos = best_pos.copy()
        new_angles = best_angles.copy()
        new_pos[idx] += rng.normal(0, step_size, size=2)
        new_angles[idx] += rng.normal(0, step_size * 3)
        
        if _is_valid_packing(new_pos, new_angles):
            # Refine outer angle with finer grid
            best_local_side = best_side
            best_local_outer = best_outer_angle
            for outer_a in np.linspace(max(0, best_outer_angle - 5), min(60, best_outer_angle + 5), 30):
                side = _compute_outer_side(new_pos, new_angles, outer_a)
                if side < best_local_side:
                    best_local_side = side
                    best_local_outer = outer_a
            
            if best_local_side < best_side - 1e-10:
                best_side = best_local_side
                best_pos = new_pos
                best_angles = new_angles
                best_outer_angle = best_local_outer
                step_size = max(0.005, step_size * 0.93)
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails > 25:
                    step_size = min(0.15, step_size * 1.05)
                    consecutive_fails = 0
    
    return best_side, best_pos, best_angles, best_outer_angle

def _optimize_single_config(pos_init, angles_init, deadline):
    """
    Optimize a single configuration using multiple phases.
    """
    best_side = float('inf')
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    best_outer_angle = 0.0
    
    # Phase 1: Grid search over global rotation and outer angle
    n_rot = 30
    n_outer = 30
    rot_range = np.linspace(0, 60, n_rot, endpoint=False)
    outer_range = np.linspace(0, 60, n_outer, endpoint=False)
    
    for rot in rot_range:
        if time.time() > deadline - 80:
            break
        angles = np.full(11, rot)
        if not _is_valid_packing(pos_init, angles):
            continue
        for outer_a in outer_range:
            side = _compute_outer_side(pos_init, angles, outer_a)
            if side < best_side:
                best_side = side
                best_pos = pos_init.copy()
                best_angles = angles.copy()
                best_outer_angle = outer_a
    
    if best_side == float('inf'):
        if _is_valid_packing(pos_init, angles_init):
            best_side = _compute_outer_side(pos_init, angles_init, 0)
            best_outer_angle = 0.0
    
    if best_side >= float('inf'):
        return best_side, best_pos, best_angles, best_outer_angle
    
    # Phase 2: Random perturbation local search
    rng = np.random.default_rng(42)
    if time.time() < deadline - 150:
        side, pos, angles, outer_a = _local_search_random(
            best_pos, best_angles, deadline - 80, rng, max_iters=4000
        )
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
            best_outer_angle = outer_a
    
    # Phase 3: Fine refinement
    if time.time() < deadline - 40:
        side, pos, angles, outer_a = _refine_fine(
            best_pos, best_angles, best_outer_angle, deadline - 15, rng, max_iters=2500
        )
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
            best_outer_angle = outer_a
    
    # Phase 4: Refined outer angle search (very fine)
    if time.time() < deadline - 10:
        for outer_a in np.linspace(max(0, best_outer_angle - 8), min(60, best_outer_angle + 8), 80):
            side = _compute_outer_side(best_pos, best_angles, outer_a)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_a
    
    # Phase 5: Rigid body rotation refinement
    if time.time() < deadline - 5:
        for rot_delta in np.linspace(-5, 5, 40):
            if time.time() > deadline - 5:
                break
            angles = np.full(11, best_angles[0] + rot_delta)
            if _is_valid_packing(best_pos, angles):
                for outer_a in np.linspace(max(0, best_outer_angle - 8), min(60, best_outer_angle + 8), 40):
                    side = _compute_outer_side(best_pos, angles, outer_a)
                    if side < best_side:
                        best_side = side
                        best_angles = angles.copy()
                        best_outer_angle = outer_a
    
    return best_side, best_pos, best_angles, best_outer_angle

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580
    
    best_side = float('inf')
    best_pos = None
    best_angles = None
    best_outer_angle = 0.0
    
    configs = _build_11_configs()
    
    # Allocate time budget per config, giving more time to early configs
    # that tend to find good solutions
    n_configs = len(configs)
    total_time = deadline - start_time - 40  # safety margin
    # Weighted time allocation: earlier configs get slightly more time
    weights = np.linspace(1.2, 0.8, n_configs)
    weights = weights / weights.sum()
    time_per_config = total_time * weights
    
    for cfg_idx, (pos_init, angles_init) in enumerate(configs):
        cfg_deadline = start_time + sum(time_per_config[:cfg_idx + 1])
        if cfg_deadline > deadline - 15:
            cfg_deadline = deadline - 15
        
        try:
            side, pos, angles, outer_a = _optimize_single_config(pos_init, angles_init, cfg_deadline)
            if side < best_side:
                best_side = side
                best_pos = pos
                best_angles = angles
                best_outer_angle = outer_a
        except Exception:
            continue
    
    # Fallback
    if best_pos is None or best_side >= float('inf'):
        sqrt3 = np.sqrt(3)
        pos_valid = np.zeros((11, 2))
        pos_valid[0] = [0, 0]
        pos_valid[1] = [2, 0]
        pos_valid[2] = [1, sqrt3]
        pos_valid[3] = [-1, sqrt3]
        pos_valid[4] = [-2, 0]
        pos_valid[5] = [-1, -sqrt3]
        pos_valid[6] = [1, -sqrt3]
        pos_valid[7] = [4, 0]
        pos_valid[8] = [-4, 0]
        pos_valid[9] = [2, 3*sqrt3]
        pos_valid[10] = [-2, 3*sqrt3]
        best_pos = pos_valid
        best_angles = np.zeros(11)
        best_outer_angle = 0.0
        best_side = _compute_outer_side(best_pos, best_angles, best_outer_angle)
    
    # Final validation
    if not _is_valid_packing(best_pos, best_angles):
        sqrt3 = np.sqrt(3)
        pos_valid = np.zeros((11, 2))
        pos_valid[0] = [0, 0]
        pos_valid[1] = [2, 0]
        pos_valid[2] = [1, sqrt3]
        pos_valid[3] = [-1, sqrt3]
        pos_valid[4] = [-2, 0]
        pos_valid[5] = [-1, -sqrt3]
        pos_valid[6] = [1, -sqrt3]
        pos_valid[7] = [5, 0]
        pos_valid[8] = [-5, 0]
        pos_valid[9] = [2.5, 4*sqrt3/2]
        pos_valid[10] = [-2.5, 4*sqrt3/2]
        best_pos = pos_valid
        best_angles = np.zeros(11)
        best_outer_angle = 0.0
        best_side = _compute_outer_side(best_pos, best_angles, best_outer_angle)
    
    # Final outer angle fine-tuning
    if time.time() < deadline - 3:
        for outer_a in np.linspace(max(0, best_outer_angle - 3), min(60, best_outer_angle + 3), 100):
            side = _compute_outer_side(best_pos, best_angles, outer_a)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_a
    
    inner_hex_data = np.column_stack([best_pos, best_angles])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

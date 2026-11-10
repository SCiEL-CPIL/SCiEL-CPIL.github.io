import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons are separated using all 12 edge normals."""
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _is_valid_packing(positions, angles):
    """Check if all pairs of hexagons are separated."""
    verts_list = []
    for i in range(11):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    for i in range(11):
        for j in range(i+1, 11):
            if not _hex_separated(verts_list[i], verts_list[j]):
                return False
    return True

def _compute_outer_side(positions, angles, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (centered at origin)
    with the given angle that contains all inner hexagon vertices.
    """
    all_verts = []
    for i in range(11):
        v = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    a = np.radians(outer_angle)
    max_abs_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
    
    return 2.0 * max_abs_proj / np.sqrt(3)

def _build_11_configs():
    """
    Build multiple valid 11-hexagon configurations.
    """
    sqrt3 = np.sqrt(3)
    configs = []
    
    # Config 1: Hex-7 cluster + 4 at corners
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
    
    # Config 2: More compact placement
    pos2 = np.zeros((11, 2))
    pos2[0] = [0, 0]
    pos2[1] = [2, 0]
    pos2[2] = [1, sqrt3]
    pos2[3] = [-1, sqrt3]
    pos2[4] = [-2, 0]
    pos2[5] = [-1, -sqrt3]
    pos2[6] = [1, -sqrt3]
    pos2[7] = [1.5, sqrt3 * 1.5]
    pos2[8] = [-1.5, sqrt3 * 1.5]
    pos2[9] = [1.5, -sqrt3 * 1.5]
    pos2[10] = [-1.5, -sqrt3 * 1.5]
    configs.append((pos2, np.zeros(11)))
    
    # Config 3: Spread satellites
    pos3 = np.zeros((11, 2))
    pos3[0] = [0, 0]
    pos3[1] = [2, 0]
    pos3[2] = [1, sqrt3]
    pos3[3] = [-1, sqrt3]
    pos3[4] = [-2, 0]
    pos3[5] = [-1, -sqrt3]
    pos3[6] = [1, -sqrt3]
    pos3[7] = [4, 0]
    pos3[8] = [-4, 0]
    pos3[9] = [2, 3*sqrt3/2]
    pos3[10] = [-2, 3*sqrt3/2]
    configs.append((pos3, np.zeros(11)))
    
    # Config 4: Asymmetric tight packing
    pos4 = np.zeros((11, 2))
    pos4[0] = [0, 0]
    pos4[1] = [2, 0]
    pos4[2] = [1, sqrt3]
    pos4[3] = [-1, sqrt3]
    pos4[4] = [-2, 0]
    pos4[5] = [-1, -sqrt3]
    pos4[6] = [1, -sqrt3]
    pos4[7] = [3.5, sqrt3 * 0.5]
    pos4[8] = [-3.5, sqrt3 * 0.5]
    pos4[9] = [2, sqrt3 * 2]
    pos4[10] = [-2, sqrt3 * 2]
    configs.append((pos4, np.zeros(11)))
    
    # Config 5: Rhombus-like arrangement
    pos5 = np.zeros((11, 2))
    pos5[0] = [0, 0]
    pos5[1] = [2, 0]
    pos5[2] = [1, sqrt3]
    pos5[3] = [-1, sqrt3]
    pos5[4] = [-2, 0]
    pos5[5] = [-1, -sqrt3]
    pos5[6] = [1, -sqrt3]
    pos5[7] = [4, sqrt3]
    pos5[8] = [-4, -sqrt3]
    pos5[9] = [2, -sqrt3 * 2]
    pos5[10] = [-2, sqrt3 * 2]
    configs.append((pos5, np.zeros(11)))
    
    # Config 6: Staggered ring
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
    
    return configs

def _local_search_random(pos_init, angles_init, deadline, rng, max_iters=5000):
    """
    Random perturbation local search.
    """
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    best_side = _compute_outer_side(best_pos, best_angles, 0.0)
    best_outer_angle = 0.0
    
    # First, find best outer angle for initial config
    for outer_a in np.linspace(0, 60, 30):
        side = _compute_outer_side(best_pos, best_angles, outer_a)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_a
    
    step_size = 0.2
    n_improvements = 0
    consecutive_fails = 0
    
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
            # Small angle perturbation
            new_angles[idx] += rng.normal(0, step_size * 5)
        
        if _is_valid_packing(new_pos, new_angles):
            # Find best outer angle for this config
            for outer_a in np.linspace(max(0, best_outer_angle - 10), min(60, best_outer_angle + 10), 15):
                side = _compute_outer_side(new_pos, new_angles, outer_a)
                if side < best_side - 1e-9:
                    best_side = side
                    best_pos = new_pos
                    best_angles = new_angles
                    best_outer_angle = outer_a
                    n_improvements += 1
                    consecutive_fails = 0
                    step_size = max(0.02, step_size * 0.95)
                    break
            else:
                consecutive_fails += 1
                if consecutive_fails > 20:
                    step_size = min(0.5, step_size * 1.05)
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
    
    step_size = 0.05
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        idx = rng.integers(0, 11)
        new_pos = best_pos.copy()
        new_angles = best_angles.copy()
        new_pos[idx] += rng.normal(0, step_size, size=2)
        new_angles[idx] += rng.normal(0, step_size * 3)
        
        if _is_valid_packing(new_pos, new_angles):
            # Refine outer angle
            for outer_a in np.linspace(max(0, best_outer_angle - 3), min(60, best_outer_angle + 3), 20):
                side = _compute_outer_side(new_pos, new_angles, outer_a)
                if side < best_side - 1e-10:
                    best_side = side
                    best_pos = new_pos
                    best_angles = new_angles
                    best_outer_angle = outer_a
                    step_size = max(0.005, step_size * 0.92)
                    break
            else:
                step_size = min(0.15, step_size * 1.03)
    
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
    n_rot = 24
    n_outer = 24
    rot_range = np.linspace(0, 60, n_rot, endpoint=False)
    outer_range = np.linspace(0, 60, n_outer, endpoint=False)
    
    for rot in rot_range:
        if time.time() > deadline - 60:
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
    if time.time() < deadline - 120:
        side, pos, angles, outer_a = _local_search_random(
            best_pos, best_angles, deadline - 60, rng, max_iters=3000
        )
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
            best_outer_angle = outer_a
    
    # Phase 3: Fine refinement
    if time.time() < deadline - 30:
        side, pos, angles, outer_a = _refine_fine(
            best_pos, best_angles, best_outer_angle, deadline - 10, rng, max_iters=2000
        )
        if side < best_side:
            best_side = side
            best_pos = pos
            best_angles = angles
            best_outer_angle = outer_a
    
    # Phase 4: Refined outer angle search
    if time.time() < deadline - 10:
        for outer_a in np.linspace(max(0, best_outer_angle - 5), min(60, best_outer_angle + 5), 50):
            side = _compute_outer_side(best_pos, best_angles, outer_a)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_a
    
    # Phase 5: Rigid body rotation refinement
    if time.time() < deadline - 5:
        for rot_delta in np.linspace(-3, 3, 30):
            if time.time() > deadline - 5:
                break
            angles = np.full(11, best_angles[0] + rot_delta)
            if _is_valid_packing(best_pos, angles):
                for outer_a in np.linspace(max(0, best_outer_angle - 5), min(60, best_outer_angle + 5), 30):
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
    
    # Allocate time budget per config
    n_configs = len(configs)
    time_per_config = (deadline - start_time - 30) / n_configs
    
    for cfg_idx, (pos_init, angles_init) in enumerate(configs):
        cfg_deadline = start_time + (cfg_idx + 1) * time_per_config
        if cfg_deadline > deadline - 10:
            cfg_deadline = deadline - 10
        
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
    
    inner_hex_data = np.column_stack([best_pos, best_angles])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

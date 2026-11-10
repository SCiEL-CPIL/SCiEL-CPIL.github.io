# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def _compute_overlap_penalty(inner_data, s, outer_angle_deg):
    """
    Compute total penalty from overlaps and boundary violations.
    Returns (total_penalty, overlap_count, boundary_violation_count)
    """
    # Compute all inner vertices
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
    
    outer_verts = _hex_vertices(0, 0, s, outer_angle_deg)
    
    total_penalty = 0.0
    overlap_count = 0
    boundary_violation_count = 0
    
    # Check pair-wise separation
    for i in range(12):
        for j in range(i + 1, 12):
            # Check if hexagons overlap using SAT
            separated = False
            for verts in (inner_verts_list[i], inner_verts_list[j]):
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    proj1 = inner_verts_list[i] @ n
                    proj2 = inner_verts_list[j] @ n
                    if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                overlap_count += 1
                # Estimate overlap depth: minimum separation needed
                # Use center distance as proxy
                c1 = inner_data[i, :2]
                c2 = inner_data[j, :2]
                dist = np.linalg.norm(c1 - c2)
                min_dist = 2.0 * math.sqrt(3)  # minimum center distance for non-overlap
                if dist < min_dist:
                    total_penalty += (min_dist - dist) ** 2
    
    # Check boundary containment
    for i in range(12):
        for v in inner_verts_list[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    boundary_violation_count += 1
                    total_penalty += (-cross) ** 2
    
    return total_penalty, overlap_count, boundary_violation_count

def _physics_relax(inner_data, outer_angle_deg, max_iters=2000, deadline=None):
    """
    Physics-based relaxation: minimize overlap penalty by adjusting positions.
    Uses gradient descent with adaptive step size.
    """
    inner = inner_data.copy()
    best_inner = inner.copy()
    best_s = float('inf')
    best_penalty = float('inf')
    
    # Compute initial s
    all_verts = _get_all_inner_vertices(inner)
    current_s = _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle_deg)
    best_s = current_s
    
    # Initial penalty
    penalty, _, _ = _compute_overlap_penalty(inner, current_s, outer_angle_deg)
    best_penalty = penalty
    
    step_size = 0.5
    lr = 0.1
    no_improve = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces (gradients) for each hexagon
        forces = np.zeros((12, 2))
        
        # Pair-wise repulsion
        for i in range(12):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            for j in range(i + 1, 12):
                vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
                
                # Check if overlapping
                separated = False
                for verts in (vi, vj):
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k + 1) % 6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / (np.linalg.norm(n) + 1e-15)
                        proj1 = vi @ n
                        proj2 = vj @ n
                        if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                            separated = True
                            break
                    if separated:
                        break
                
                if not separated:
                    # Overlapping - apply repulsive force
                    c1 = inner[i, :2]
                    c2 = inner[j, :2]
                    diff = c1 - c2
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        dir_vec = diff / dist
                        # Force proportional to overlap
                        force_mag = 1.0
                        forces[i] += force_mag * dir_vec
                        forces[j] -= force_mag * dir_vec
        
        # Boundary constraint force
        outer_verts = _hex_vertices(0, 0, current_s, outer_angle_deg)
        for i in range(12):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            for v in vi:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < -1e-6:
                        # Vertex outside - push inward
                        # Normal of edge (inward)
                        edge = p2 - p1
                        n = np.array([edge[1], -edge[0]])
                        n = n / (np.linalg.norm(n) + 1e-15)
                        # Check if this is inward normal
                        center = np.array([0.0, 0.0])
                        edge_mid = (p1 + p2) / 2
                        if np.dot(n, center - edge_mid) < 0:
                            n = -n
                        forces[i] += 0.5 * n * (-cross)
        
        # Apply forces with adaptive step
        trial_inner = inner.copy()
        trial_inner[:, :2] += lr * forces
        
        # Recompute s for trial
        trial_all_verts = _get_all_inner_vertices(trial_inner)
        trial_s = _min_s_for_all_vertices(trial_all_verts, np.array([0, 0]), outer_angle_deg)
        
        # Compute trial penalty
        trial_penalty, trial_overlaps, trial_boundary = _compute_overlap_penalty(trial_inner, trial_s, outer_angle_deg)
        
        # Accept if penalty decreased or if it's a small step
        if trial_penalty < best_penalty or (trial_penalty < penalty and trial_overlaps <= 12):
            inner = trial_inner
            current_s = trial_s
            penalty = trial_penalty
            no_improve = 0
            if trial_penalty < best_penalty:
                best_penalty = trial_penalty
                best_inner = inner.copy()
                best_s = trial_s
            # Increase learning rate slightly
            lr = min(lr * 1.05, 1.0)
        else:
            no_improve += 1
            # Decrease learning rate
            lr = max(lr * 0.9, 0.01)
        
        if no_improve > 50:
            break
    
    return best_inner, best_s, best_penalty

def _objective_scipy(x, outer_angle_deg):
    """
    Objective function for scipy optimizer.
    x: (24,) array of 12 (x,y) positions.
    Returns: s value (to be minimized).
    """
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = x[0:12]
    inner_data[:, 1] = x[12:24]
    inner_data[:, 2] = 0.0  # All inner hexagons have angle 0
    
    all_verts = _get_all_inner_vertices(inner_data)
    s = _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle_deg)
    
    # Add penalty for overlaps
    penalty, overlap_count, _ = _compute_overlap_penalty(inner_data, s, outer_angle_deg)
    
    return s + penalty

def _scipy_optimize(inner_data, outer_angle_deg, deadline):
    """
    Use scipy.optimize to minimize the objective.
    """
    x0 = np.zeros(24)
    x0[0:12] = inner_data[:, 0]
    x0[12:24] = inner_data[:, 1]
    
    result = minimize(
        _objective_scipy,
        x0,
        args=(outer_angle_deg,),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
    )
    
    # Extract result
    inner_data_opt = np.zeros((12, 3))
    inner_data_opt[:, 0] = result.x[0:12]
    inner_data_opt[:, 1] = result.x[12:24]
    inner_data_opt[:, 2] = 0.0
    
    all_verts = _get_all_inner_vertices(inner_data_opt)
    s_opt = _min_s_for_all_vertices(all_verts, np.array([0, 0]), outer_angle_deg)
    
    return inner_data_opt, s_opt, result.fun

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Direct construction: use a compact triangular lattice arrangement
    dx_row = 3.0  # horizontal spacing
    dy_row = 3 * math.sqrt(3) / 2  # vertical spacing
    dx_offset = 1.5  # horizontal offset for alternating rows
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Try multiple grid arrangements
    arrangements = [
        (4, 3),  # 4 rows, 3 cols
        (3, 4),  # 3 rows, 4 cols
        (2, 6),  # 2 rows, 6 cols
        (6, 2),  # 6 rows, 2 cols
    ]
    
    for n_rows, n_cols in arrangements:
        positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * dx_row
                y = row * dy_row
                if row % 2 == 1:
                    x += dx_offset
                positions.append([x, y, 0.0])
        
        inner = np.array(positions)
        com = np.mean(inner[:, :2], axis=0)
        inner[:, :2] -= com
        
        all_verts = _get_all_inner_vertices(inner)
        
        for angle_deg in np.linspace(0, 60, 61):
            s_cand = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
            if s_cand < best_s:
                best_s = s_cand
                best_inner = inner.copy()
                best_outer = np.array([0, 0, angle_deg])
    
    # Now do physics-based relaxation from the best starting point
    if best_inner is not None:
        current_inner = best_inner.copy()
        current_outer = best_outer.copy()
        current_s = best_s
        
        # Try physics relaxation from multiple angles
        for angle_deg in np.linspace(0, 60, 13):
            if time.time() > deadline:
                break
            
            # Start from current best and relax
            relaxed_inner, relaxed_s, _ = _physics_relax(
                current_inner.copy(), angle_deg, 
                max_iters=1500, deadline=deadline
            )
            
            if relaxed_s < current_s:
                # Validate
                trial_outer = np.array([0, 0, angle_deg])
                if _is_valid(relaxed_inner, trial_outer, relaxed_s):
                    current_inner = relaxed_inner
                    current_outer = trial_outer
                    current_s = relaxed_s
        
        # Use scipy optimization for further refinement
        for angle_deg in np.linspace(0, 60, 7):
            if time.time() > deadline:
                break
            
            # Start from current best and use scipy
            scipy_inner, scipy_s, _ = _scipy_optimize(
                current_inner.copy(), angle_deg, deadline
            )
            
            if scipy_s < current_s:
                # Validate
                trial_outer = np.array([0, 0, angle_deg])
                if _is_valid(scipy_inner, trial_outer, scipy_s):
                    current_inner = scipy_inner
                    current_outer = trial_outer
                    current_s = scipy_s
        
        # Final simulated annealing polish
        np.random.seed(42)
        step_size = 0.3
        T = 0.5
        no_improve_count = 0
        max_no_improve = 20
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(0, 12)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            trial_inner = current_inner.copy()
            trial_inner[idx, 0] += dx
            trial_inner[idx, 1] += dy
            
            # Check separation
            valid = True
            trial_verts = _hex_vertices(trial_inner[idx, 0], trial_inner[idx, 1], 1.0, trial_inner[idx, 2])
            for j in range(12):
                if j == idx:
                    continue
                other_verts = _hex_vertices(trial_inner[j, 0], trial_inner[j, 1], 1.0, trial_inner[j, 2])
                if not _check_separation(trial_verts, other_verts):
                    valid = False
                    break
            
            if valid:
                trial_all_verts = _get_all_inner_vertices(trial_inner)
                best_trial_s = float('inf')
                best_trial_angle = current_outer[2]
                for angle_offset in np.linspace(-3, 3, 7):
                    trial_angle = current_outer[2] + angle_offset
                    s_cand = _min_s_for_all_vertices(trial_all_verts, np.array([0, 0]), trial_angle)
                    if s_cand < best_trial_s:
                        best_trial_s = s_cand
                        best_trial_angle = trial_angle
                
                delta_s = current_s - best_trial_s
                if delta_s > 0:
                    trial_outer = np.array([0, 0, best_trial_angle])
                    if _is_valid(trial_inner, trial_outer, best_trial_s):
                        current_inner = trial_inner
                        current_outer = trial_outer
                        current_s = best_trial_s
                        no_improve_count = 0
                        T = max(T * 0.99, 0.01)
                else:
                    if np.random.random() < math.exp(-delta_s / T):
                        trial_outer = np.array([0, 0, best_trial_angle])
                        if _is_valid(trial_inner, trial_outer, best_trial_s):
                            current_inner = trial_inner
                            current_outer = trial_outer
                            current_s = best_trial_s
                            T = max(T * 0.99, 0.01)
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                step_size *= 0.85
                no_improve_count = 0
                T *= 0.85
                if step_size < 0.01:
                    break
        
        best_inner = current_inner
        best_outer = current_outer
        best_s = current_s
    
    # Final validation with fallback
    if best_inner is None or not _is_valid(best_inner, best_outer, best_s):
        # Use a simple valid fallback
        inner_fallback = np.array([
            [0, 0, 0],
            [-3, 0, 0],
            [3, 0, 0],
            [-1.5, 2.598, 0],
            [1.5, 2.598, 0],
            [-4.5, 2.598, 0],
            [4.5, 2.598, 0],
            [-1.5, -2.598, 0],
            [1.5, -2.598, 0],
            [-4.5, -2.598, 0],
            [4.5, -2.598, 0],
            [0, -5.196, 0]
        ])
        com_fallback = np.mean(inner_fallback[:, :2], axis=0)
        inner_fallback[:, :2] -= com_fallback
        best_inner = inner_fallback
        best_outer = np.array([0, 0, 0])
        best_s = 10.0
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END

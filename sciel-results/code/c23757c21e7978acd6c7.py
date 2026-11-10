# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = np.zeros((12, 6, 2))
    for i in range(12):
        verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], side, angles_deg[i])
    return verts

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2) or (12,6,2)),
    compute minimum outer hex side s (centered at origin, with given angle)
    that contains all vertices.
    """
    if vertices_all.ndim == 3:
        vertices_all = vertices_all.reshape(-1, 2)
    
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=50.0):
    """Smooth approximation of _min_outer_side using log-sum-exp."""
    if vertices_all.ndim == 3:
        vertices_all = vertices_all.reshape(-1, 2)
    
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (72, 6)
    
    beta_dots = beta * dots
    max_beta_dots = np.max(beta_dots)
    if max_beta_dots > 700:
        return (2.0 / np.sqrt(3.0)) * (max_beta_dots / beta)
    soft_max = max_beta_dots / beta + np.log(np.sum(np.exp(beta_dots - max_beta_dots))) / beta
    
    if soft_max < 0:
        soft_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * soft_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using vectorized SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    
    all_verts = np.zeros((n, 6, 2))
    all_normals = np.zeros((n, 6, 2))
    
    for i in range(n):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], side, angles_deg[i])
        all_verts[i] = verts
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e + 1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx * nx + ny * ny)
            if norm > 1e-12:
                all_normals[i, e] = [nx / norm, ny / norm]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = all_verts[i]
            v2 = all_verts[j]
            overlap = True
            for hex_idx in range(2):
                normals = all_normals[i] if hex_idx == 0 else all_normals[j]
                proj1 = v1 @ normals.T  # (6, 6)
                proj2 = v2 @ normals.T  # (6, 6)
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                separated = (max1 + tol < min2) | (max2 + tol < min1)
                if np.any(separated):
                    overlap = False
                    break
            if overlap:
                return False
    return True

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    # SAT check
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            separated = False
            for k in range(6):
                edge = verts_i[(k + 1) % 6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                        separated = True
                        break
            if not separated:
                for k in range(6):
                    edge = verts_j[(k + 1) % 6] - verts_j[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                            separated = True
                            break
            if not separated:
                return False
    # Containment
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _initial_honeycomb():
    """Create initial honeycomb-like arrangement of 12 hexagons."""
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _objective(params, outer_angle_deg=0.0):
    """Objective: minimize outer hex side s, with penalty for overlaps."""
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _smooth_objective(params, outer_angle_deg=0.0, beta=50.0):
    """Smooth objective for L-BFGS-B using soft max."""
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _scipy_polish(params, deadline, outer_angle_deg=0.0, beta=50.0):
    """Use scipy L-BFGS-B to polish the solution with smooth objective."""
    def obj(x):
        return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success or result.fun < _smooth_objective(params, outer_angle_deg=outer_angle_deg, beta=beta):
            if _check_nonoverlap(result.x.reshape(12, 3)[:, :2], result.x.reshape(12, 3)[:, 2]):
                return result.x
    except Exception:
        pass
    return params

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    """Perform local refinement using simulated annealing and scipy polish."""
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    # Phase 1: Simulated annealing with small steps
    np.random.seed(42)
    T = 0.05
    cooling = 0.999
    step_size = 0.05
    
    current_params = params.copy()
    current_s = s
    
    for it in range(30000):
        if time.time() > deadline - 30:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < s:
                s = s_new
                params = new_params.copy()
        
        T *= cooling
        if it % 1000 == 0:
            step_size *= 0.9
    
    # Phase 2: Scipy polish with smooth objective
    if time.time() < deadline - 10:
        polished = _scipy_polish(params, deadline, outer_angle_deg, beta=50.0)
        if _check_nonoverlap(polished.reshape(12, 3)[:, :2], polished.reshape(12, 3)[:, 2]):
            verts_polished = _compute_all_vertices(polished.reshape(12, 3)[:, :2], polished.reshape(12, 3)[:, 2])
            s_polished = _min_outer_side(verts_polished, outer_angle_deg)
            if s_polished < s:
                params = polished
                s = s_polished
    
    return params, s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Multiple initial configurations
    configs = []
    
    # Honeycomb arrangement
    pos_h, ang_h = _initial_honeycomb()
    configs.append((pos_h.copy(), ang_h.copy()))
    
    # Seed arrangement
    seed_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    seed_angles = np.zeros(12)
    configs.append((seed_positions.copy(), seed_angles.copy()))
    
    # Hexagonal grid arrangement
    hex_positions = np.array([
        [0, 0],
        [3, 0],
        [1.5, sqrt3],
        [-1.5, sqrt3],
        [-3, 0],
        [-1.5, -sqrt3],
        [1.5, -sqrt3],
        [4.5, sqrt3],
        [3, 2*sqrt3],
        [0, 2*sqrt3],
        [-3, 2*sqrt3],
        [-4.5, sqrt3]
    ])
    hex_angles = np.zeros(12)
    configs.append((hex_positions.copy(), hex_angles.copy()))
    
    # Rotated hex grid
    for rot in [15, 30, 45]:
        ang = np.full(12, rot)
        configs.append((hex_positions.copy(), ang))
    
    # Perturbed hex grid
    for t in range(20):
        np.random.seed(t * 100 + 42)
        pos = hex_positions + np.random.randn(12, 2) * 0.1
        ang = np.random.uniform(-30, 30, 12)
        configs.append((pos, ang))
    
    best_params = None
    best_s = float('inf')
    
    for idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        # Quick Nelder-Mead
        try:
            res = minimize(_objective, p, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Local refinement with SA + polish
        if time.time() < deadline - 60:
            refined_params, refined_s = _local_refinement(p, deadline)
            if refined_s < best_s:
                best_s = refined_s
                best_params = refined_params.copy()
    
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = seed_positions[i, 0]
            best_params[3*i+1] = seed_positions[i, 1]
            best_params[3*i+2] = seed_angles[i]
        best_s = _min_outer_side(_compute_all_vertices(seed_positions, seed_angles))
    
    # Final refinement
    if time.time() < deadline - 10:
        try:
            res_final = minimize(_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res_final.fun < best_s:
                best_s = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
        
        # Another polish pass
        if time.time() < deadline - 5:
            polished = _scipy_polish(best_params, deadline, beta=100.0)
            if _check_nonoverlap(polished.reshape(12, 3)[:, :2], polished.reshape(12, 3)[:, 2]):
                verts_polished = _compute_all_vertices(polished.reshape(12, 3)[:, :2], polished.reshape(12, 3)[:, 2])
                s_polished = _min_outer_side(verts_polished)
                if s_polished < best_s:
                    best_params = polished
                    best_s = s_polished
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[3*i]
        inner_hex_data[i, 1] = best_params[3*i+1]
        inner_hex_data[i, 2] = best_params[3*i+2]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    centers = np.column_stack([best_params[0::3], best_params[1::3]])
    angles_deg = best_params[2::3]
    s = _min_outer_side(_compute_all_vertices(centers, angles_deg))
    
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END

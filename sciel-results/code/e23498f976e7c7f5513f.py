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

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
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
    """
    Smooth approximation of _min_outer_side using log-sum-exp.
    """
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
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        all_verts[i] = verts
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                all_normals[i, e] = [nx/norm, ny/norm]
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = all_verts[i]
            v2 = all_verts[j]
            overlap = True
            for hex_idx in range(2):
                normals = all_normals[i] if hex_idx == 0 else all_normals[j]
                proj1 = v1 @ normals.T
                proj2 = v2 @ normals.T
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

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
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

def _objective(params, side=1.0, outer_angle_deg=0.0):
    """
    Objective: minimize outer hex side s, with penalty for overlaps.
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    if not _check_nonoverlap(centers, angles_deg, side):
        return s + 50.0
    
    return s

def _smooth_objective(params, side=1.0, outer_angle_deg=0.0, beta=50.0):
    """
    Smooth objective for L-BFGS-B using soft max.
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    if not _check_nonoverlap(centers, angles_deg, side):
        return s + 50.0
    
    return s

def _scipy_polish(params, deadline, outer_angle_deg=0.0, beta=50.0):
    """
    Use scipy L-BFGS-B to polish the solution with smooth objective.
    """
    def obj(x):
        if time.time() > deadline - 5:
            return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
        return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < _smooth_objective(params, outer_angle_deg=outer_angle_deg, beta=beta):
            new_params = result.x
            if _check_nonoverlap(new_params.reshape(12,3)[:, :2], new_params.reshape(12,3)[:, 2]):
                return new_params
    except Exception:
        pass
    return params

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    """
    Perform local refinement using simulated annealing and scipy polish.
    """
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    current_params = params.copy()
    current_s = s
    best_params = params.copy()
    best_s = s
    
    # Phase 1: Simulated annealing
    T = 0.05
    cooling = 0.9995
    step_size = 0.05
    rng = np.random.default_rng(42)
    
    for it in range(20000):
        if time.time() > deadline - 30:
            break
        
        new_params = current_params + rng.standard_normal(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        if it % 500 == 0:
            step_size *= 0.95
    
    # Phase 2: Scipy polish
    if time.time() < deadline - 15:
        polished = _scipy_polish(best_params, deadline, outer_angle_deg, beta=50.0)
        if polished is not None:
            p2d = polished.reshape(12, 3)
            if _check_nonoverlap(p2d[:, :2], p2d[:, 2]):
                verts2 = _compute_all_vertices(p2d[:, :2], p2d[:, 2])
                s2 = _min_outer_side(verts2, outer_angle_deg)
                if s2 < best_s:
                    best_s = s2
                    best_params = polished
    
    return best_params, best_s

def _search_outer_angle(params, deadline):
    """
    Search over outer hexagon angle to find the best orientation.
    """
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    all_verts = _compute_all_vertices(centers, angles_deg)
    
    best_s = np.inf
    best_angle = 0.0
    
    # Coarse search
    for a in np.linspace(0, 60, 121):
        s = _min_outer_side(all_verts, a)
        if s < best_s:
            best_s = s
            best_angle = a
    
    # Fine search around best
    for a in np.linspace(best_angle - 5, best_angle + 5, 201):
        s = _min_outer_side(all_verts, a)
        if s < best_s:
            best_s = s
            best_angle = a
    
    # Even finer
    for a in np.linspace(best_angle - 0.5, best_angle + 0.5, 101):
        s = _min_outer_side(all_verts, a)
        if s < best_s:
            best_s = s
            best_angle = a
    
    return best_angle, best_s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Initial positions
    positions, angles_deg = _initial_honeycomb()
    
    # Verify non-overlap
    if not _check_nonoverlap(positions, angles_deg):
        # Fallback
        positions = np.array([
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
            [0, -4.0],
        ], dtype=float)
        angles_deg = np.zeros(12)
    
    # Convert to params format
    params = np.column_stack([positions, angles_deg]).flatten()
    
    # Phase 1: Local refinement
    params, s = _local_refinement(params, deadline, outer_angle_deg=0.0)
    
    # Phase 2: Search outer angle
    if time.time() < deadline - 10:
        outer_angle, s = _search_outer_angle(params, deadline)
    
    # Phase 3: Final refinement with best outer angle
    if time.time() < deadline - 20:
        params, s = _local_refinement(params, deadline, outer_angle_deg=outer_angle)
        outer_angle, s = _search_outer_angle(params, deadline)
    
    # Final verification
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    if not _check_nonoverlap(centers, angles_deg):
        # Fallback to seed
        centers = np.array([
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
            [0, -4.0],
        ], dtype=float)
        angles_deg = np.zeros(12)
        outer_angle = 0.0
    
    # Recompute s precisely
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle)
    
    inner_hex_data = np.column_stack([centers, angles_deg])
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END

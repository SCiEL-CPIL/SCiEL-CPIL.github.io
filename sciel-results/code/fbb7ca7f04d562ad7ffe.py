# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_edge_normals_vec(angle_deg):
    """Return 6 unit outward edge normals for a hexagon with given angle."""
    a = np.radians(angle_deg)
    angles = a + np.radians(30) + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (N,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots)
    
    if max_dot < 0:
        max_dot = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * max_dot
    return s_min

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=80.0):
    """
    Smooth approximation of _min_outer_side using log-sum-exp.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (N, 6)
    flat = dots.ravel()
    
    beta_dots = beta * flat
    max_beta_dots = np.max(beta_dots)
    if max_beta_dots > 500:
        return (2.0 / np.sqrt(3.0)) * (max_beta_dots / beta)
    
    soft_max = max_beta_dots / beta + np.log(np.sum(np.exp(beta_dots - max_beta_dots))) / beta
    
    if soft_max < 0:
        soft_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * soft_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
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
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                all_normals[i, e] = [nx/norm, ny/norm]
    
    for i in range(n):
        for j in range(i+1, n):
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

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all vertices of all hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _honeycomb_seed():
    """
    Create a honeycomb-like arrangement of 12 hexagons.
    1 center + 6 ring + 5 outer = 12
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring: 5 hexagons (skipping one to make 12 total)
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
    """
    Objective: minimize outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 100.0
    
    return s

def _smooth_objective(params, outer_angle_deg=0.0, beta=80.0):
    """
    Smooth objective for L-BFGS-B using soft max.
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 100.0
    
    return s

def _scipy_polish(params, outer_angle_deg=0.0, beta=80.0, maxiter=300, maxfun=3000):
    """
    Use scipy L-BFGS-B to polish the solution with smooth objective.
    """
    def obj(x):
        return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < _smooth_objective(params, outer_angle_deg=outer_angle_deg, beta=beta):
            # Verify with hard check
            c = result.x.reshape(12, 3)[:, :2]
            a = result.x.reshape(12, 3)[:, 2]
            if _check_nonoverlap(c, a):
                return result.x
    except Exception:
        pass
    return params

def _directed_sa_move(params, centers, angles_deg, outer_angle_deg, step_size=0.05):
    """
    Perform a directed single-hexagon move: pick one hexagon and nudge it.
    Returns new_params if accepted, else None.
    """
    idx = np.random.randint(12)
    new_params = params.copy()
    
    # Nudge position
    new_params[idx*3] += np.random.randn() * step_size
    new_params[idx*3+1] += np.random.randn() * step_size
    # Small angle perturbation
    new_params[idx*3+2] += np.random.randn() * step_size * 0.5
    
    c_new = new_params.reshape(12, 3)[:, :2]
    a_new = new_params.reshape(12, 3)[:, 2]
    
    if not _check_nonoverlap(c_new, a_new):
        return None
    
    verts_new = _compute_all_vertices(c_new, a_new)
    s_new = _min_outer_side(verts_new, outer_angle_deg)
    
    return new_params, s_new

def _simulated_annealing(params, outer_angle_deg, deadline, max_iters=50000):
    """
    Simulated annealing with directed single-hexagon moves.
    """
    T = 0.08
    cooling = 0.9995
    step_size = 0.06
    
    current_params = params.copy()
    c = current_params.reshape(12, 3)[:, :2]
    a = current_params.reshape(12, 3)[:, 2]
    all_verts = _compute_all_vertices(c, a)
    current_s = _min_outer_side(all_verts, outer_angle_deg)
    
    best_params = current_params.copy()
    best_s = current_s
    
    for it in range(max_iters):
        if time.time() > deadline - 40:
            break
        
        result = _directed_sa_move(current_params, c, a, outer_angle_deg, step_size)
        if result is None:
            continue
        
        new_params, s_new = result
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            c = current_params.reshape(12, 3)[:, :2]
            a = current_params.reshape(12, 3)[:, 2]
            
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        if it % 2000 == 0:
            step_size *= 0.95
            if step_size < 0.005:
                step_size = 0.005
    
    return best_params, best_s

def _rigid_cluster_optimization(outer_angle_deg, deadline):
    """
    Optimize a rigid cluster: fix relative positions, only optimize global rotation.
    This reduces 36 variables to 1 (rotation angle).
    """
    positions, angles = _honeycomb_seed()
    
    def cluster_objective(theta_rad):
        # Rotate all positions by theta_rad
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        rot_positions = np.column_stack([
            positions[:, 0] * cos_t - positions[:, 1] * sin_t,
            positions[:, 0] * sin_t + positions[:, 1] * cos_t
        ])
        # Also rotate angles
        rot_angles = angles + np.degrees(theta_rad)
        
        all_verts = _compute_all_vertices(rot_positions, rot_angles)
        s = _min_outer_side(all_verts, outer_angle_deg)
        return s
    
    # Optimize rotation angle
    result = minimize(cluster_objective, 0.0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    theta_opt = result.x[0]
    cos_t = np.cos(theta_opt)
    sin_t = np.sin(theta_opt)
    rot_positions = np.column_stack([
        positions[:, 0] * cos_t - positions[:, 1] * sin_t,
        positions[:, 0] * sin_t + positions[:, 1] * cos_t
    ])
    rot_angles = angles + np.degrees(theta_opt)
    
    # Build params
    params = np.zeros(36)
    for i in range(12):
        params[i*3] = rot_positions[i, 0]
        params[i*3+1] = rot_positions[i, 1]
        params[i*3+2] = rot_angles[i]
    
    return params

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses SA + L-BFGS-B with directed moves and rigid cluster initialization.
    """
    deadline = time.time() + 600.0
    
    best_params = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Phase 1: Try rigid cluster optimization for different outer angles
    for oa_deg in [0.0, 15.0, 30.0, 45.0]:
        if time.time() > deadline - 120:
            break
        try:
            params = _rigid_cluster_optimization(oa_deg, deadline)
            c = params.reshape(12, 3)[:, :2]
            a = params.reshape(12, 3)[:, 2]
            if _check_nonoverlap(c, a):
                verts = _compute_all_vertices(c, a)
                s = _min_outer_side(verts, oa_deg)
                if s < best_s:
                    best_s = s
                    best_params = params.copy()
                    best_outer_angle = oa_deg
        except Exception:
            pass
    
    # Phase 2: Honeycomb seed + SA + L-BFGS-B for multiple outer angles
    for oa_deg in [0.0, 15.0, 30.0, 45.0]:
        if time.time() > deadline - 60:
            break
        
        positions, angles = _honeycomb_seed()
        params = np.zeros(36)
        for i in range(12):
            params[i*3] = positions[i, 0]
            params[i*3+1] = positions[i, 1]
            params[i*3+2] = angles[i]
        
        # SA phase
        sa_deadline = min(deadline - 90, time.time() + 45)
        if time.time() < sa_deadline:
            params, sa_s = _simulated_annealing(params, oa_deg, sa_deadline)
        
        # L-BFGS-B polish
        if time.time() < deadline - 30:
            params = _scipy_polish(params, oa_deg, beta=80.0, maxiter=400, maxfun=4000)
        
        c = params.reshape(12, 3)[:, :2]
        a = params.reshape(12, 3)[:, 2]
        if _check_nonoverlap(c, a):
            verts = _compute_all_vertices(c, a)
            s = _min_outer_side(verts, oa_deg)
            if s < best_s:
                best_s = s
                best_params = params.copy()
                best_outer_angle = oa_deg
    
    # Phase 3: Additional L-BFGS-B restarts from best solution
    if best_params is not None:
        for restart in range(8):
            if time.time() > deadline - 30:
                break
            
            # Perturb best solution
            rng = np.random.default_rng(restart + 1000)
            perturbed = best_params + rng.normal(0, 0.02, size=36)
            
            polished = _scipy_polish(perturbed, best_outer_angle, beta=80.0, maxiter=300, maxfun=3000)
            
            c = polished.reshape(12, 3)[:, :2]
            a = polished.reshape(12, 3)[:, 2]
            if _check_nonoverlap(c, a):
                verts = _compute_all_vertices(c, a)
                s = _min_outer_side(verts, best_outer_angle)
                if s < best_s:
                    best_s = s
                    best_params = polished.copy()
    
    # Build final result
    if best_params is None:
        # Fallback: honeycomb
        positions, angles = _honeycomb_seed()
        best_params = np.zeros(36)
        for i in range(12):
            best_params[i*3] = positions[i, 0]
            best_params[i*3+1] = positions[i, 1]
            best_params[i*3+2] = angles[i]
        c = best_params.reshape(12, 3)[:, :2]
        a = best_params.reshape(12, 3)[:, 2]
        verts = _compute_all_vertices(c, a)
        best_s = _min_outer_side(verts, 0.0)
        best_outer_angle = 0.0
    
    # Ensure s >= sqrt(12)
    best_s = max(best_s, math.sqrt(12))
    
    inner = best_params.reshape(12, 3)
    outer = np.array([0.0, 0.0, best_outer_angle])
    
    return inner, outer, best_s

# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    hull = ConvexHull(all_verts)
    hull_verts = all_verts[hull.vertices]
    return hull_verts

def _objective_smooth(params, centers):
    """
    params: [outer_cx, outer_cy, outer_angle_deg, phi_0_deg, ..., phi_10_deg]
    Returns smooth approximation of enclosing hexagon side length.
    Uses log-sum-exp for smooth max.
    """
    outer_cx, outer_cy, outer_angle = params[0], params[1], params[2]
    phis = params[3:]
    
    # Compute all vertices
    all_verts = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], phis[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Get outer normals
    normals = _get_outer_normals(outer_angle)
    
    # Translate relative to outer center
    rel_verts = all_verts - np.array([outer_cx, outer_cy])
    
    # Project onto all 6 normals
    dots = rel_verts @ normals.T  # shape (N*6, 6)
    
    # For each normal direction, we need max(|dot|)
    # Use soft-max for smoothness
    beta = 50.0  # Sharpness parameter
    abs_dots = np.abs(dots)
    
    # For each of the 6 directions, compute soft max of |dots|
    # Then take max over directions
    soft_maxes = []
    for k in range(6):
        vals = abs_dots[:, k]
        # Log-sum-exp for stability
        max_val = np.max(vals)
        log_sum = max_val + np.log(np.sum(np.exp(vals - max_val)))
        soft_maxes.append(log_sum)
    
    soft_max_arr = np.array(soft_maxes)
    max_val = np.max(soft_max_arr)
    total_log_sum = max_val + np.log(np.sum(np.exp(soft_max_arr - max_val)))
    
    S = 2 * total_log_sum / np.sqrt(3)
    return S

def _objective_exact(params, centers):
    """
    params: [outer_cx, outer_cy, outer_angle_deg, phi_0_deg, ..., phi_10_deg]
    Returns exact enclosing hexagon side length.
    """
    outer_cx, outer_cy, outer_angle = params[0], params[1], params[2]
    phis = params[3:]
    
    angles = phis
    composite = _compute_composite_hull(centers, angles)
    S = _min_enclosing_hex_side(composite, outer_angle, np.array([outer_cx, outer_cy]))
    return S

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Flower packing: 1 center + 6 ring + 4 outer
    eps = 1e-9
    centers_list = []
    angles_list = []
    
    # Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(0.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(0.0)
    
    # Outer ring: 4 hexagons at distance 3 from center, at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(0.0)
        
    centers = np.array(centers_list)
    n_hex = len(centers)
    
    # Parameters: [outer_cx, outer_cy, outer_angle, phi_0, ..., phi_10]
    # All angles in degrees
    best_S = float('inf')
    best_params = np.zeros(n_hex + 3)
    
    # Try multiple initial configurations
    initial_configs = [
        # [outer_cx, outer_cy, outer_angle, phis...]
        [0.0, 0.0, 0.0] + [0.0] * n_hex,
        [0.0, 0.0, 30.0] + [0.0] * n_hex,
        [0.0, 0.0, 15.0] + [0.0] * n_hex,
        [0.0, 0.0, 45.0] + [0.0] * n_hex,
        [0.0, 0.0, 0.0] + [30.0] * n_hex,
        [0.0, 0.0, 30.0] + [30.0] * n_hex,
    ]
    
    # Also try some random inits
    rng = np.random.RandomState(42)
    for _ in range(4):
        cfg = [0.0, 0.0, rng.uniform(0, 60)] + [rng.uniform(0, 60) for _ in range(n_hex)]
        initial_configs.append(cfg)
    
    # Phase 1: Use smooth objective for faster convergence
    for cfg in initial_configs:
        if time.time() > DEADLINE - 30.0:
            break
        
        x0 = np.array(cfg)
        try:
            res = minimize(_objective_smooth, x0, args=(centers,), method='Powell',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            
            if time.time() > DEADLINE - 15.0:
                break
                
            # Evaluate with exact objective
            exact_S = _objective_exact(res.x, centers)
            if exact_S < best_S:
                best_S = exact_S
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with exact objective using Nelder-Mead
    if time.time() < DEADLINE - 10.0:
        try:
            res = minimize(_objective_exact, best_params, args=(centers,), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < DEADLINE - 5.0:
        try:
            res = minimize(_objective_exact, best_params, args=(centers,), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Extract results
    outer_cx, outer_cy, outer_angle = best_params[0], best_params[1], best_params[2]
    phis = best_params[3:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], phis[i]])
    
    outer_center = (outer_cx, outer_cy)
    side = best_S + 1e-9  # Tiny margin for safety
    
    return inner, outer_center, side, outer_angle

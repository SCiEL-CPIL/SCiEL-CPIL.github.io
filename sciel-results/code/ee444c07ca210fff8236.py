import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Precompute triangle indices for 13 points
_N = 13
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)

def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull

def _soft_min(areas, beta):
    """Smooth approximation of the minimum area."""
    # To avoid overflow/underflow in exp, we shift by max
    max_area = np.max(areas)
    # We want -1/beta * log(sum(exp(-beta * a)))
    # This is equivalent to -1/beta * log(sum(exp(-beta * (a - max)))) - max
    # Wait, the standard softmin is -1/beta * log(sum(exp(-beta * x)))
    # Let's use the stable version:
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    # = -1/beta * ( -beta * max(x) + log(sum(exp(-beta * (x - max(x))))) )
    # = max(x) - 1/beta * log(sum(exp(-beta * (x - max(x)))))
    # This is numerically stable.
    shifted = areas - max_area
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    return max_area - (1.0 / beta) * log_sum_exp

def _objective_neg_smooth(params, beta):
    """
    Objective function for optimization.
    Returns negative score to be minimized.
    params: flattened array of point coordinates.
    """
    pts = params.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    a_hull = _hull_area(pts)
    
    if a_hull <= 0 or not np.isfinite(a_hull):
        return 1e10
    
    # We want to maximize A_min / A_hull
    # Using softmin for A_min
    a_min_soft = _soft_min(areas, beta)
    
    score = a_min_soft / a_hull
    if not np.isfinite(score):
        return 1e10
        
    return -score

def _generate_symmetric_config(radii, angles, center):
    """
    Generate a symmetric configuration from parameters.
    radii: list of radii for each ring
    angles: list of angles for points in each ring
    center: (x, y) for the center point (if any)
    """
    pts = []
    if center is not None:
        pts.append(center)
    
    for r, ang_list in zip(radii, angles):
        for ang in ang_list:
            pts.append([r * np.cos(ang), r * np.sin(ang)])
            
    return np.array(pts)

def _optimize_symmetric_topology(topology, start_time, time_limit):
    """
    Optimize a specific symmetric topology.
    topology: dict defining the structure
    """
    # Define parameters to optimize:
    # For each ring: radius
    # For each ring: phase offset (rotation)
    # Center: x, y (if present)
    
    n_rings = topology['n_rings']
    n_points_per_ring = topology['n_points_per_ring']
    has_center = topology['has_center']
    
    # Parameter vector:
    # [r_1, r_2, ..., r_n, phi_1, phi_2, ..., phi_n, cx, cy]
    n_params = n_rings * 2 + (2 if has_center else 0)
    
    def unpack(params):
        radii = params[:n_rings]
        phases = params[n_rings:2*n_rings]
        center = None
        if has_center:
            center = params[2*n_rings:2*n_rings+2]
        
        pts = []
        if has_center:
            pts.append(center)
            
        for i in range(n_rings):
            r = radii[i]
            phi = phases[i]
            n_pts = n_points_per_ring[i]
            # Regular polygon angles
            base_angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
            ring_angles = base_angles + phi
            for ang in ring_angles:
                pts.append([r * np.cos(ang), r * np.sin(ang)])
                
        return np.array(pts)

    def objective(params):
        if time.time() - start_time > time_limit - 5:
            return 1e10
        pts = unpack(params)
        if not np.all(np.isfinite(pts)):
            return 1e10
        return _objective_neg_smooth(pts, beta=50.0) # Fixed beta for stability, refine later if needed

    # Initial guess: Regular polygon-like
    best_result = None
    best_score = 1e10
    
    # Try a few initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Radii between 0.5 and 1.5
        init_radii = rng.uniform(0.5, 1.5, n_rings)
        # Phases random
        init_phases = rng.uniform(0, 2*np.pi, n_rings)
        # Center near 0
        init_center = rng.normal(0, 0.1, 2) if has_center else []
        
        init_params = np.concatenate([init_radii, init_phases, init_center])
        
        try:
            res = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        return None, 0.0
        
    # Convert back to points and calculate exact score
    final_pts = unpack(best_result)
    # Normalize to unit hull area for consistent comparison? 
    # The objective function handles scaling implicitly via the ratio.
    # However, L-BFGS-B might drift in scale. 
    # Since we are maximizing A_min/A_hull, the scale cancels out.
    # But numerical stability might benefit from normalization.
    # Let's just return the points. The exact score will be calculated later.
    
    score = _exact_score(final_pts)
    return final_pts, score

def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses symmetry-constrained optimization with L-BFGS-B on a smoothed objective.
    """
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Define candidate topologies for 13 points
    # 1. 1 center + 6 inner + 6 outer
    topologies = [
        {'n_rings': 2, 'n_points_per_ring': [6, 6], 'has_center': True},
        # 2. 1 center + 12 outer
        {'n_rings': 1, 'n_points_per_ring': [12], 'has_center': True},
        # 3. 1 center + 4 + 8
        {'n_rings': 2, 'n_points_per_ring': [4, 8], 'has_center': True},
        # 4. 1 center + 3 + 4 + 5
        {'n_rings': 3, 'n_points_per_ring': [3, 4, 5], 'has_center': True},
        # 5. 13 on one ring (regular 13-gon)
        {'n_rings': 1, 'n_points_per_ring': [13], 'has_center': False},
        # 6. 1 center + 5 + 7
        {'n_rings': 2, 'n_points_per_ring': [5, 7], 'has_center': True},
    ]
    
    for topo in topologies:
        if time.time() - start_time > time_limit - 60:
            break
            
        pts, score = _optimize_symmetric_topology(topo, start_time, time_limit)
        if pts is not None and score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # If no good result from symmetric optimization, fall back to a known good configuration
    # e.g., Regular 13-gon
    if best_pts is None or best_score < 1.0: # Regular 13-gon score is approx 0.13
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _exact_score(best_pts)
        
    # Final Polish: Use Nelder-Mead on the exact (non-smooth) objective from the best symmetric result
    # This helps correct for the bias of the soft-min approximation
    if best_pts is not None and time.time() - start_time < time_limit - 30:
        def obj_exact(x):
            p = x.reshape(13, 2)
            s = _exact_score(p)
            if s <= 0 or not np.isfinite(s):
                return 1e10
            return -s
            
        try:
            res = minimize(
                obj_exact,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.all(np.isfinite(res.x)):
                polished_pts = res.x.reshape(13, 2)
                polished_score = _exact_score(polished_pts)
                if polished_score > best_score:
                    best_pts = polished_pts
                    best_score = polished_score
        except Exception:
            pass

    # Fallback check
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END

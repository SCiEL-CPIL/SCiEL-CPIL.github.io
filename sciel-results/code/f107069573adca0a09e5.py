import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective_soft(points_flat, beta=100.0):
    """
    Soft-min objective for smooth optimization.
    Minimizes the soft-min of triangle areas, normalized by hull area.
    """
    pts = points_flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Check for degenerate configurations
    if np.min(_all_areas(pts)) < 1e-10:
        return 1e10
        
    areas = _all_areas(pts)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, use log-sum-exp trick
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    
    if soft_min < 1e-10:
        return 1e10
        
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min

def _objective_hard(points_flat):
    pts = points_flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Strategy 1: Optimize the 1+6+7 symmetric structure
    # Parameters: r1, r2, 7 angles for the outer ring
    # Total 9 parameters
    
    def params_to_points_167(params):
        r1 = params[0]
        r2 = params[1]
        angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        pts_inner = np.column_stack([r1 * np.cos(angles_inner), r1 * np.sin(angles_inner)])
        
        angles_outer = params[2:9]
        # Normalize angles to [0, 2pi) to avoid issues, though cos/sin handles it
        pts_outer = np.column_stack([r2 * np.cos(angles_outer), r2 * np.sin(angles_outer)])
        
        pts = np.vstack([[0.0, 0.0], pts_inner, pts_outer])
        return pts

    def obj_167(params):
        pts = params_to_points_167(params)
        # Use soft objective for smoothness
        return _objective_soft(pts.flatten(), beta=50.0)

    # Initial guess for 1+6+7
    # r1 ~ 1, r2 ~ 2
    # Angles outer: evenly spaced
    x0_167 = np.array([1.0, 2.0])
    angles_outer_0 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    x0_167 = np.concatenate([x0_167, angles_outer_0])
    
    if time.time() < deadline - 20.0:
        try:
            res = minimize(obj_167, x0_167, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success or res.fun < 1e9:
                pts = params_to_points_167(res.x)
                score = _score(pts)
                update_best(pts, score)
        except Exception:
            pass

    # Strategy 2: Optimize Regular 14-gon with radial perturbations
    # This is harder to parameterize simply, so we use full 28D search
    # but initialized with a regular polygon
    
    def obj_full(flat):
        return _objective_soft(flat, beta=50.0)

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    if time.time() < deadline - 15.0:
        try:
            res = minimize(obj_full, x0_reg, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success or res.fun < 1e9:
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
        except Exception:
            pass

    # Strategy 3: General local search from best found so far
    if best_points is not None and time.time() < deadline - 10.0:
        # Try Nelder-Mead for final polish
        try:
            res = minimize(_objective_hard, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            update_best(pts, score)
        except Exception:
            pass

    # Strategy 4: Multi-start with random perturbations of the best structure
    if best_points is not None and time.time() < deadline - 5.0:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 5.0:
                break
            np.random.seed(seed)
            noise = np.random.normal(0, 0.1, (N, 2))
            x0 = best_points + noise
            try:
                res = minimize(obj_full, x0.flatten(), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
            except Exception:
                pass

    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points

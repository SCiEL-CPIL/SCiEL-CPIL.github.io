import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools
import warnings

N = 13
DEADLINE = 140.0 # Safety margin for 150s limit

# Precompute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def _compute_triangle_areas(pts):
    """
    Vectorized computation of areas for all triples of points.
    """
    p1 = pts[TRI_INDICES[:, 0]]
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """
    Compute the area of the convex hull of the points.
    Uses scipy.spatial.ConvexHull for robustness.
    """
    try:
        hull = __import__('scipy.spatial', fromlist=['ConvexHull']).ConvexHull(pts)
        return hull.volume # In 2D, volume is area
    except Exception:
        # Fallback: shoelace on sorted points if hull fails (unlikely for valid inputs)
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]
        x = pts_sorted[:, 0]
        y = pts_sorted[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _softmin_score(pts, beta):
    """
    Compute the score using soft-min of triangle areas.
    """
    areas = _compute_triangle_areas(pts)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Shift for stability
    a_min = np.min(areas)
    shifted = -beta * (areas - a_min)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = a_min - log_sum / beta
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return soft_min_area / hull_area

def _neg_softmin_score(params, beta):
    pts = params.reshape(N, 2)
    return -_softmin_score(pts, beta)

def _neg_score(params):
    pts = params.reshape(N, 2)
    return -_score(pts)

def _make_ring_config(r_inner, r_outer, n_inner, n_outer, phase=0.0):
    """
    Create a configuration with points on two concentric rings.
    n_inner points on circle radius r_inner.
    n_outer points on circle radius r_outer.
    Total N = n_inner + n_outer.
    """
    if n_inner + n_outer != N:
        raise ValueError("n_inner + n_outer must equal N")
    
    pts = []
    if n_inner > 0:
        angles_in = 2 * np.pi * np.arange(n_inner) / n_inner + phase
        pts.extend(np.column_stack([r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)]))
    if n_outer > 0:
        angles_out = 2 * np.pi * np.arange(n_outer) / n_outer + phase
        pts.extend(np.column_stack([r_outer * np.cos(angles_out), r_outer * np.sin(angles_out)]))
        
    return np.array(pts)

def _optimize_from_start(start_pts, deadline):
    """
    Optimize from a starting point using annealed soft-min.
    """
    current_params = start_pts.flatten()
    best_score = _score(start_pts)
    best_params = current_params.copy()
    
    # Annealing schedule for beta
    betas = [5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        try:
            result = minimize(
                lambda x: _neg_softmin_score(x, beta),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_score:
                candidate_pts = result.x.reshape(N, 2)
                s_true = _score(candidate_pts)
                if s_true > best_score:
                    best_score = s_true
                    best_params = result.x.copy()
                current_params = result.x
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on the hard score
    if time.time() < deadline - 2:
        try:
            result_nm = minimize(
                _neg_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            if result_nm.fun < -best_score:
                best_score = -result_nm.fun
                best_params = result_nm.x
        except Exception:
            pass
            
    return best_params.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Ring configurations
    # Try different splits of points between inner and outer rings
    # and different radius ratios
    for n_inner in range(0, N):
        n_outer = N - n_inner
        if n_inner == 0 or n_outer == 0:
            continue
        # Radius ratios: inner/outer
        for ratio in [0.3, 0.5, 0.7, 0.9]:
            # Normalize so outer radius is 1
            r_out = 1.0
            r_in = ratio
            # Add a small phase shift to break symmetry if needed
            phase = 0.0 if n_inner % 2 == 0 else np.pi / n_inner
            try:
                pts = _make_ring_config(r_in, r_out, n_inner, n_outer, phase)
                inits.append(pts)
            except Exception:
                pass
                
    # 3. Random convex configurations
    # Generate random points, take convex hull, then jitter
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.rand(N, 2)
        # Ensure they are roughly convex by sorting by angle and checking
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]
        inits.append(pts_sorted)
        
    best_points = None
    best_score = -np.inf
    
    # Optimize from each diverse start
    for i, init_pts in enumerate(inits):
        if time.time() > DEADLINE - 10:
            break
            
        # Check if this init is already good enough to skip? No, always try to improve.
        # But we can skip if time is tight and we have a good score.
        
        try:
            pts_opt, score_opt = _optimize_from_start(init_pts, DEADLINE)
            if score_opt > best_score:
                best_score = score_opt
                best_points = pts_opt
        except Exception:
            pass
            
    # If no optimization succeeded, fallback to best initial
    if best_points is None:
        best_points = inits[0]
        
    return best_points

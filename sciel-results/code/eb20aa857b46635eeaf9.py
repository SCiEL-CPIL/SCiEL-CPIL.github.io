import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns a smooth approximation of the minimum area.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp
    max_a = np.max(areas)
    # Numerical stability: shift by max
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _neg_soft_score(points_flat, beta=100.0):
    """
    Negative soft-score for gradient-based optimization.
    This is the objective function for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e10 # Large penalty for degenerate hull
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 1e10

def _neg_hard_score(points_flat):
    """
    Negative hard score for non-gradient methods like Nelder-Mead.
    """
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_points(r_inner, delta_inner, jitter_o, jitter_i):
    """
    Generate 14 points with flexible angular positions on two rings.
    """
    n_ring = 7
    base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    
    outer_angles = base_angles + jitter_o
    inner_angles = base_angles + delta_inner + jitter_i
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # 1. Generate diverse initial points
    initial_points = []
    
    # Regular 14-gon
    initial_points.append(_regular_polygon(14, 1.0, 0.0))
    initial_points.append(_regular_polygon(14, 1.0, np.pi/14.0))
    
    # Two-ring structures
    rng = np.random.RandomState(42)
    for r in [0.3, 0.5, 0.7, 0.8]:
        for delta in [0.0, np.pi/7.0]:
            pts = _generate_two_ring_points(r, delta, np.zeros(7), np.zeros(7))
            initial_points.append(pts)
            
            # Add some jittered versions
            jitter_o = rng.normal(0, 0.1, 7)
            jitter_i = rng.normal(0, 0.1, 7)
            pts_j = _generate_two_ring_points(r, delta, jitter_o, jitter_i)
            initial_points.append(pts_j)
            
    # Random convex points (ensure convexity by sorting by angle from centroid)
    for _ in range(10):
        # Generate random points in a disk
        radii = np.sqrt(rng.uniform(0.1, 1.0, 14))
        angles = np.sort(rng.uniform(0, 2*np.pi, 14))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_points.append(pts)

    # 2. Optimize each initial point
    for init_pts in initial_points:
        if _time_left() < 5:
            break
            
        init_flat = init_pts.flatten()
        
        # Step A: Coarse local optimization with Nelder-Mead on hard score
        # This helps settle into a good basin for the soft score
        try:
            res_nm = scipy.optimize.minimize(
                _neg_hard_score,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            current_flat = res_nm.x
        except Exception:
            current_flat = init_flat

        # Step B: Progressive refinement with L-BFGS-B on soft score
        # Start with low beta (smooth) and increase to high beta (sharp)
        betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if _time_left() < 2:
                break
                
            try:
                res_lbfgs = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, beta),
                    current_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_flat = res_lbfgs.x
                
                # Check hard score
                pts_check = _normalize(current_flat.reshape(-1, 2))
                sc_check = _score(pts_check)
                _update_best(pts_check, sc_check)
                
            except Exception:
                pass

    # 3. If time remains, do a few random restarts from best point with noise
    if best_points is not None and _time_left() > 10:
        for _ in range(20):
            if _time_left() < 2:
                break
            noise = rng.normal(0, 0.05, 28)
            perturbed = best_points.flatten() + noise
            try:
                res_polish = scipy.optimize.minimize(
                    _neg_soft_score,
                    perturbed,
                    args=(100.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

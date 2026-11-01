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

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft-score for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

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
    if max_a < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return soft_min

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

def _generate_three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    """
    Generate 14 points as union of three concentric regular polygons.
    Sizes: n1, n2, n3 (sum = 14)
    Radii: r1, r2, r3
    Rotations: rot1, rot2, rot3
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_score_three_ring(params):
    """
    Negative score for three-ring parameterization.
    params: [r1, r2, r3, rot1, rot2, rot3]
    """
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        
        pts = _generate_three_ring(7, 5, 2, r1, r2, r3, rot1, rot2, rot3)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_score_three_ring_662(params):
    """
    Negative score for three-ring parameterization (6,6,2 split).
    params: [r1, r2, r3, rot1, rot2, rot3]
    """
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        
        pts = _generate_three_ring(6, 6, 2, r1, r2, r3, rot1, rot2, rot3)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_soft_score_three_ring(params, beta=100.0):
    """
    Negative soft-score for three-ring parameterization (for L-BFGS-B).
    """
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        
        pts = _generate_three_ring(7, 5, 2, r1, r2, r3, rot1, rot2, rot3)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_soft_score_three_ring_662(params, beta=100.0):
    """
    Negative soft-score for three-ring parameterization (6,6,2 split) (for L-BFGS-B).
    """
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        
        pts = _generate_three_ring(6, 6, 2, r1, r2, r3, rot1, rot2, rot3)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _regular_14_gon():
    return _regular_polygon(14, 1.0, 0.0)

def _two_ring_7_7(r, delta):
    """Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r)."""
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_two_ring_7_7(params):
    """Negative score for two-ring 7/7 parameterization."""
    r, delta = params
    if r <= 0:
        r = 1e-6
    pts = _two_ring_7_7(r, delta)
    pts_norm = _normalize(pts)
    return -_score(pts_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # 1. Coarse grid search on two-ring 7/7 parameterization
    r_vals = np.linspace(0.1, 0.95, 20)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 15, endpoint=False)
    
    best_two_ring = None
    best_two_ring_score = 0.0
    
    for r in r_vals:
        for delta in delta_vals:
            sc = -_neg_score_two_ring_7_7([r, delta])
            if sc > best_two_ring_score:
                best_two_ring_score = sc
                best_two_ring = [r, delta]
    
    if best_two_ring is not None:
        pts_tr = _two_ring_7_7(best_two_ring[0], best_two_ring[1])
        _update_best(pts_tr, best_two_ring_score)
    
    if _time_left() < 30:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 2. Refine two-ring 7/7 with Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            _neg_score_two_ring_7_7,
            best_two_ring,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        pts_opt = _two_ring_7_7(res.x[0], res.x[1])
        sc_opt = _score(_normalize(pts_opt))
        _update_best(pts_opt, sc_opt)
    except Exception:
        pass

    if _time_left() < 25:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 3. Three-ring 7/5/2 optimization using differential evolution
    bounds_752 = [(0.1, 1.5), (0.1, 1.5), (0.1, 1.5), 
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    try:
        de_res = scipy.optimize.differential_evolution(
            _neg_score_three_ring,
            bounds=bounds_752,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=True
        )
        pts_de = _generate_three_ring(7, 5, 2, 
                                       de_res.x[0], de_res.x[1], de_res.x[2],
                                       de_res.x[3], de_res.x[4], de_res.x[5])
        sc_de = _score(_normalize(pts_de))
        _update_best(pts_de, sc_de)
    except Exception:
        pass

    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 4. Three-ring 6/6/2 optimization using differential evolution
    bounds_662 = [(0.1, 1.5), (0.1, 1.5), (0.1, 1.5), 
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    try:
        de_res_662 = scipy.optimize.differential_evolution(
            _neg_score_three_ring_662,
            bounds=bounds_662,
            seed=43,
            maxiter=100,
            tol=1e-8,
            polish=True
        )
        pts_de_662 = _generate_three_ring(6, 6, 2, 
                                           de_res_662.x[0], de_res_662.x[1], de_res_662.x[2],
                                           de_res_662.x[3], de_res_662.x[4], de_res_662.x[5])
        sc_de_662 = _score(_normalize(pts_de_662))
        _update_best(pts_de_662, sc_de_662)
    except Exception:
        pass

    if _time_left() < 15:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 5. Polish best three-ring result with L-BFGS-B using soft surrogate
    if best_points is not None:
        try:
            # Determine which configuration is best by checking structure
            # For simplicity, try polishing both 7/5/2 and 6/6/2
            for split, func_soft, func_exact in [(752, _neg_soft_score_three_ring, _neg_score_three_ring),
                                                  (662, _neg_soft_score_three_ring_662, _neg_score_three_ring_662)]:
                if _time_left() < 10:
                    break
                
                # Initialize from a reasonable starting point
                if split == 752:
                    init = np.array([1.0, 0.5, 0.2, 0.0, 0.0, 0.0])
                else:
                    init = np.array([1.0, 0.6, 0.3, 0.0, 0.0, 0.0])
                
                for beta in [50.0, 100.0, 200.0, 500.0]:
                    if _time_left() < 5:
                        break
                    res_lbfgs = scipy.optimize.minimize(
                        lambda x, b=beta: func_soft(x, b),
                        init,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if split == 752:
                        pts_lbfgs = _generate_three_ring(7, 5, 2, 
                                                          res_lbfgs.x[0], res_lbfgs.x[1], res_lbfgs.x[2],
                                                          res_lbfgs.x[3], res_lbfgs.x[4], res_lbfgs.x[5])
                    else:
                        pts_lbfgs = _generate_three_ring(6, 6, 2, 
                                                          res_lbfgs.x[0], res_lbfgs.x[1], res_lbfgs.x[2],
                                                          res_lbfgs.x[3], res_lbfgs.x[4], res_lbfgs.x[5])
                    sc_lbfgs = _score(_normalize(pts_lbfgs))
                    _update_best(pts_lbfgs, sc_lbfgs)
                    init = res_lbfgs.x
        except Exception:
            pass

    if _time_left() < 10:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 6. Full 28D local optimization from best point
    if best_points is not None:
        try:
            res_full = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            pts_full = _normalize(res_full.x.reshape(-1, 2))
            sc_full = _score(pts_full)
            _update_best(pts_full, sc_full)
        except Exception:
            pass

    if _time_left() < 5:
        return best_points if best_points is not None else _normalize(_regular_14_gon())

    # 7. Final polish with L-BFGS-B on full 28D using soft surrogate
    if best_points is not None and _time_left() > 3:
        try:
            for beta in [50.0, 100.0, 200.0]:
                if _time_left() < 3:
                    break
                res_polish = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
        except Exception:
            pass

    if best_points is None:
        best_points = _normalize(_regular_14_gon())
    else:
        best_points = _normalize(best_points)
        
    return best_points

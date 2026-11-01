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
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp (numerically stable)
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(np.unique(points, axis=0)) < 3:
            return 0.0
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

def _neg_score(points_flat):
    """Negative score for minimization."""
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

def _generate_7_7_regular(r, delta):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r).
    Regular heptagons with relative rotation delta.
    """
    n = 7
    angles_outer = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _generate_7_7_free(params):
    """
    params: [r, theta_0..6, phi_0..6]
    Outer: unit circle, angles theta.
    Inner: radius r, angles phi.
    """
    r = params[0]
    theta = params[1:8]
    phi = params[8:15]
    
    points_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    points_inner = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    
    return np.vstack([points_outer, points_inner])

def _neg_soft_score_7_7_free(params, beta=100.0):
    try:
        points = _generate_7_7_free(params)
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_score_7_7_free(params):
    try:
        points = _generate_7_7_free(params)
        return -_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Phase 1: Coarse Search in Parameter Space (7+7 Regular)
    # Optimize r and delta
    def _neg_soft_score_7_7_reg(params, beta=50.0):
        r, delta = params
        if r < 0.01: r = 0.01
        if r > 1.0: r = 1.0
        pts = _generate_7_7_regular(r, delta)
        ha = _hull_area(pts)
        if ha < 1e-12: return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    
    # Initial guesses for r, delta
    # r in [0.3, 0.8], delta in [0, 2pi/7]
    best_reg_params = np.array([0.5, 0.0])
    
    for r_init in [0.4, 0.6, 0.8]:
        for d_init in [0.0, np.pi/7, 2*np.pi/7]:
            if _time_left() < 20: break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=50.0: _neg_soft_score_7_7_reg(x, b),
                    np.array([r_init, d_init]),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                sc = _score(_generate_7_7_regular(res.x[0], res.x[1]))
                if sc > best_score:
                    _update_best(_generate_7_7_regular(res.x[0], res.x[1]), sc)
                    best_reg_params = res.x
            except Exception:
                pass

    # Phase 2: Fine Search in Parameter Space (7+7 Free Angles)
    # Initialize from best regular solution
    r_best, d_best = best_reg_params
    theta_init = np.linspace(0, 2*np.pi, 7, endpoint=False)
    phi_init = np.linspace(d_best, d_best + 2*np.pi, 7, endpoint=False)
    params_init = np.concatenate(([r_best], theta_init, phi_init))
    
    # Optimize free angles
    for beta in [50.0, 100.0, 200.0]:
        if _time_left() < 15: break
        try:
            res = scipy.optimize.minimize(
                lambda x, b=beta: _neg_soft_score_7_7_free(x, b),
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_pts = _generate_7_7_free(res.x)
            sc_new = _score(new_pts)
            if sc_new > best_score:
                _update_best(new_pts, sc_new)
                params_init = res.x
        except Exception:
            pass
            
    # Nelder-Mead polish on free params
    if _time_left() > 10:
        try:
            res_nm = scipy.optimize.minimize(
                _neg_score_7_7_free,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            new_pts = _generate_7_7_free(res_nm.x)
            sc_new = _score(new_pts)
            if sc_new > best_score:
                _update_best(new_pts, sc_new)
        except Exception:
            pass

    # Phase 3: Full Space Optimization (28 dims)
    if _time_left() > 25 and best_points is not None:
        current_pts = best_points.copy()
        
        for beta in [100.0, 300.0]:
            if _time_left() < 10: break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
                    current_pts = new_pts
            except Exception:
                pass
        
        # Final NM polish
        if _time_left() > 10:
            try:
                res_nm = scipy.optimize.minimize(
                    _neg_score,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_pts = _normalize(res_nm.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    # Phase 4: Random Restarts / Perturbations
    if _time_left() > 15 and best_points is not None:
        rng = np.random.RandomState(42)
        base_pts = best_points
        for _ in range(5):
            if _time_left() < 8: break
            noise = rng.normal(0, 0.1, base_pts.shape)
            init_pts = _normalize(base_pts + noise)
            try:
                res = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, 150.0),
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

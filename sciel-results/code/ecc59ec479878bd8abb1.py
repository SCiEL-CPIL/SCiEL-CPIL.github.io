"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    This is differentiable and guides optimizers better.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _make_symmetric_points(theta, r_inner, r_outer, n_outer, n_inner):
    """
    Construct points with rotational symmetry.
    n_outer points on a circle of radius r_outer at angles 2*pi*k/n_outer + theta
    n_inner points on a circle of radius r_inner at angles 2*pi*k/n_inner + theta + offset
    """
    if n_outer + n_inner != N:
        return None
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + theta
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    
    offset = np.pi / n_inner if n_inner > 0 else 0
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + theta + offset
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([pts_outer, pts_inner])

def _neg_symmetric_score(params):
    """
    Score for symmetric configuration.
    params: [theta, r_inner, r_outer, n_outer, n_inner]
    But n_outer, n_inner are discrete. We'll handle them externally or as continuous approximations.
    Let's fix the split for a given call and only optimize theta, r_inner, r_outer.
    """
    # This function is a placeholder. We will use a wrapper in the optimization loop.
    return 0.0

def _optimize_symmetric_split(n_outer, n_inner, init_theta=0.0, init_r_in=0.5, init_r_out=1.0):
    """Optimize a specific symmetric split."""
    def neg_score_sym(params):
        theta, r_in, r_out = params
        # Ensure radii are positive
        r_in = abs(r_in)
        r_out = abs(r_out)
        if r_in < 1e-6 or r_out < 1e-6:
            return 1.0
        pts = _make_symmetric_points(theta, r_in, r_out, n_outer, n_inner)
        if pts is None:
            return 1.0
        return -_score(pts)
    
    def neg_smooth_sym(params):
        theta, r_in, r_out = params
        r_in = abs(r_in)
        r_out = abs(r_out)
        if r_in < 1e-6 or r_out < 1e-6:
            return 1.0
        pts = _make_symmetric_points(theta, r_in, r_out, n_outer, n_inner)
        if pts is None:
            return 1.0
        return -_smooth_score(pts)
    
    x0 = np.array([init_theta, init_r_in, init_r_out])
    
    best_score = -1.0
    best_x = x0.copy()
    
    # Try Powell on smooth
    try:
        res = minimize(neg_smooth_sym, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        x_opt = res.x
        s = -neg_score_sym(x_opt)
        if s > best_score:
            best_score = s
            best_x = x_opt
    except Exception:
        pass
    
    # Try Nelder-Mead on exact
    try:
        res = minimize(neg_score_sym, best_x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        x_opt = res.x
        s = -neg_score_sym(x_opt)
        if s > best_score:
            best_score = s
            best_x = x_opt
    except Exception:
        pass
    
    return best_score, best_x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # 1. Symmetric Optimization
    # Try various splits of N=13 into outer and inner rings
    # Possible splits (outer, inner):
    # (13, 0) - just a polygon
    # (12, 1) - 12 on ring, 1 center
    # (11, 2)
    # (10, 3)
    # (9, 4)
    # (8, 5)
    # (7, 6)
    
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20.0:
            break
        
        # Try a few initializations for each split
        for init_r_in in [0.0, 0.3, 0.5, 0.7]:
            if n_in == 0:
                init_r_in = 0.0
            for init_theta in [0.0, np.pi / n_out]:
                s, x_opt = _optimize_symmetric_split(n_out, n_in, init_theta, init_r_in, 1.0)
                if s > best_score:
                    best_score = s
                    pts = _make_symmetric_points(x_opt[0], abs(x_opt[1]), abs(x_opt[2]), n_out, n_in)
                    if pts is not None:
                        best_pts = pts
    
    # 2. General Optimization from Best Symmetric
    if time.time() < deadline - 15.0:
        # Use the best symmetric points as a starting point for general optimization
        try:
            res_smooth = minimize(
                _neg_smooth_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            smooth_pts = res_smooth.x.reshape(-1, 2)
            smooth_score = _score(smooth_pts)
            if smooth_score > best_score:
                best_score = smooth_score
                best_pts = smooth_pts
        except Exception:
            pass
        
        try:
            res_exact = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            exact_pts = res_exact.x.reshape(-1, 2)
            exact_score = _score(exact_pts)
            if exact_score > best_score:
                best_score = exact_score
                best_pts = exact_pts
        except Exception:
            pass

    # 3. Random Restarts (if time permits)
    if time.time() < deadline - 30.0:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            # Random points in unit disk
            r = rng.rand(N) ** 0.5
            theta = rng.rand(N) * 2 * np.pi
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            try:
                res = minimize(
                    _neg_score,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                opt_pts = res.x.reshape(-1, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_pts = opt_pts
            except Exception:
                pass

    return best_pts

# EVOLVE-BLOCK-END

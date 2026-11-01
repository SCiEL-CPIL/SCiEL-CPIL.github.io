import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _spread_out_init(rng, n=N):
    """Initialize points by maximizing minimum pairwise distance (TSP-like)."""
    points = np.zeros((n, 2))
    # Start with a random point near center
    points[0] = rng.uniform(-0.5, 0.5, 2)
    
    for i in range(1, n):
        # Try many candidate positions, pick the one that maximizes min distance to existing points
        best_pos = None
        best_min_dist = -1
        for attempt in range(200):
            # Sample from disk
            r = np.sqrt(rng.uniform(0, 1))
            theta = rng.uniform(0, 2 * np.pi)
            candidate = np.array([r * np.cos(theta), r * np.sin(theta)])
            dists = np.linalg.norm(points[:i] - candidate, axis=1)
            min_dist = dists.min()
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = candidate
        points[i] = best_pos
    
    # Normalize to unit area hull
    ha = _hull_area(points)
    if ha > 1e-12:
        points = points / np.sqrt(ha)
    
    return points

def _random_in_disk_init(rng, n=N):
    """Random points in unit disk."""
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    ha = _hull_area(points)
    if ha > 1e-12:
        points = points / np.sqrt(ha)
    return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
        return sc

    def _optimize_from(init_points, beta_schedule=None, max_total_iter=1500):
        """Optimize from initial points using annealed smooth-min."""
        if time.time() > deadline - 10:
            return
        
        x = init_points.flatten()
        
        if beta_schedule is None:
            # Default annealing schedule
            n_stages = 8
            betas = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
        else:
            betas = beta_schedule
        
        iters_per_stage = max(50, max_total_iter // len(betas))
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': iters_per_stage, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        return x.reshape(-1, 2)

    def _coordinate_descent_polish(points, n_rounds=2):
        """Polish each point individually using Nelder-Mead."""
        pts = points.copy()
        for rnd in range(n_rounds):
            if time.time() > deadline - 5:
                break
            for i in range(N):
                def _neg_score_one_point(params):
                    new_pts = pts.copy()
                    new_pts[i, 0] = params[0]
                    new_pts[i, 1] = params[1]
                    return -_score(new_pts)
                
                x0 = np.array([pts[i, 0], pts[i, 1]])
                try:
                    res = scipy.optimize.minimize(
                        _neg_score_one_point,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        return pts

    # Phase 1: Regular polygon as baseline
    reg = _regular_polygon(N, 1.0)
    ha = _hull_area(reg)
    if ha > 1e-12:
        reg = reg / np.sqrt(ha)
    _update_best(reg)
    
    # Phase 2: Spread-out initialization (maximize min distance)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        init = _spread_out_init(rng)
        _update_best(init)
        if time.time() < deadline - 20:
            _optimize_from(init, max_total_iter=1200)
    
    # Phase 3: Random disk initialization with optimization
    for trial in range(8):
        if time.time() > deadline - 20:
            break
        init = _random_in_disk_init(rng)
        _update_best(init)
        if time.time() < deadline - 15:
            _optimize_from(init, max_total_iter=1000)
    
    # Phase 4: Perturbed regular polygon
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        noise_scale = 0.05 + 0.05 * trial
        noise = rng.uniform(-noise_scale, noise_scale, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        ha = _hull_area(init)
        if ha > 1e-12:
            init = init / np.sqrt(ha)
        _update_best(init)
        if time.time() < deadline - 12:
            _optimize_from(init, max_total_iter=1000)
    
    # Phase 5: Coordinate descent polish on best
    if best_points is not None and time.time() < deadline - 8:
        polished = _coordinate_descent_polish(best_points, n_rounds=2)
        _update_best(polished)
    
    # Phase 6: Final high-beta refinement
    if best_points is not None and time.time() < deadline - 5:
        x = best_points.flatten()
        def _obj_high_beta(x_flat):
            return _neg_smooth_score_and_grad(x_flat, 2000.0)
        try:
            res = scipy.optimize.minimize(
                _obj_high_beta,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    # Phase 7: Final coordinate descent
    if best_points is not None and time.time() < deadline - 3:
        polished = _coordinate_descent_polish(best_points, n_rounds=1)
        _update_best(polished)
    
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        ha = _hull_area(best_points)
        if ha > 1e-12:
            best_points = best_points / np.sqrt(ha)
    
    return best_points

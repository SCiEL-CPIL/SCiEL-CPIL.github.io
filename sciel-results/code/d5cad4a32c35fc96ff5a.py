import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
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

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Smooth surrogate for -score using log-sum-exp soft-min of triangle areas.
    Returns (value, gradient) for L-BFGS-B.
    """
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
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    # soft_min = max_area - (1/beta) * log(sum_exp)
    smooth_min = max_area - np.log(sum_exp) / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    # Gradient of smooth_min w.r.t. points
    # d(smooth_min)/d(area_i) = -weights_i
    # d(area_i)/d(point) via cross product
    # area_i = 0.5 * |cross_i|
    # For smoothness, we use the sign of cross to handle the abs
    signs = np.sign(cross)
    
    # d(cross_i)/d(p_i) = -(v1 + v2) = -(p_j - p_i + p_k - p_i) = -p_j - p_k + 2*p_i
    # d(cross_i)/d(p_j) = v2 = p_k - p_i
    # d(cross_i)/d(p_k) = -v1 = -(p_j - p_i) = p_i - p_j
    
    # Let's compute gradients of cross product
    # cross = (pj-x - pi-x)*(pk-y - pi-y) - (pj-y - pi-y)*(pk-x - pi-x)
    # Let dx1 = pj-x - pi-x, dy1 = pj-y - pi-y
    # Let dx2 = pk-x - pi-x, dy2 = pk-y - pi-y
    # cross = dx1*dy2 - dy1*dx2
    
    # d(cross)/d(pi-x) = -dy2 + dx2
    # d(cross)/d(pi-y) = dx1 - dy1  (wait: d/d(pi-y) of dx1*dy2 - dy1*dx2 = 0 - (-1)*dx2 = dx2? No)
    # Let's re-derive carefully.
    # x1 = pj_x - pi_x, y1 = pj_y - pi_y
    # x2 = pk_x - pi_x, y2 = pk_y - pi_y
    # cross = x1*y2 - y1*x2
    
    # d(cross)/d(pi_x) = (dx1/dpi_x)*y2 + x1*(dy2/dpi_x) - (dy1/dpi_x)*x2 - y1*(dx2/dpi_x)
    # dx1/dpi_x = -1, dy2/dpi_x = 0
    # dy1/dpi_x = 0, dx2/dpi_x = -1
    # => -y2 - y1*(-1) = y1 - y2
    
    # d(cross)/d(pi_y) = (dx1/dpi_y)*y2 + x1*(dy2/dpi_y) - (dy1/dpi_y)*x2 - y1*(dx2/dpi_y)
    # dx1/dpi_y = 0, dy2/dpi_y = 0
    # dy1/dpi_y = -1, dx2/dpi_y = 0
    # => 0 - (-1)*x2 = x2
    
    # d(cross)/d(pj_x) = (dx1/dpj_x)*y2 + x1*(dy2/dpj_x) - ...
    # dx1/dpj_x = 1, others 0
    # => y2
    
    # d(cross)/d(pj_y) = ...
    # dy1/dpj_y = 1
    # => -x2
    
    # d(cross)/d(pk_x) = ...
    # dx2/dpk_x = 1
    # => -y1
    
    # d(cross)/d(pk_y) = ...
    # dy2/dpk_y = 1
    # => x1
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v2[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # Gradient of area = 0.5 * |cross|
    # d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
    # Gradient of smooth_min = sum_i (-weights_i) * d(area_i)/d(point)
    # Gradient of score = Gradient(smooth_min) / ha
    # We want to minimize -score, so we return -grad_score
    
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

def _generate_ring_config(radii, angles):
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _two_ring_7_7(r, delta):
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(delta, delta + 2 * np.pi, 7, endpoint=False)
    all_angles = np.concatenate([angles_outer, angles_inner])
    all_radii = np.concatenate([np.ones(7), r * np.ones(7)])
    return _generate_ring_config(all_radii, all_angles)

def _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3)])
    return _generate_ring_config(all_radii, all_angles)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = 0.0
    best_points = None

    def _update_best(pts, sc=None):
        nonlocal best_score, best_points
        if sc is None:
            sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()
        return sc

    def _time_left():
        return deadline - time.time()

    def _try_optimize_smooth(init_points, maxiter_total=3000):
        """
        Refine points using smooth surrogate with increasing beta.
        """
        if _time_left() < 10:
            return
        init_norm = _normalize(init_points)
        x = init_norm.flatten().copy()
        
        n_stages = 5
        betas = [20.0, 50.0, 100.0, 200.0, 400.0]
        
        for beta in betas:
            if _time_left() < 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_total // n_stages, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final high-beta polish
        if _time_left() > 5:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 800.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                # Final exact Nelder-Mead polish
                if _time_left() > 3:
                    res = scipy.optimize.minimize(
                        _neg_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # Start with regular 14-gon as baseline
    reg14 = _regular_polygon(14, 1.0, 0.0)
    sc_reg = _update_best(reg14)

    # Phase 1: Coarse grid search on two-ring 7/7
    r_vals = np.linspace(0.2, 0.9, 15)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 10, endpoint=False)

    best_two_ring = None
    best_two_ring_score = 0.0

    for r in r_vals:
        for delta in delta_vals:
            pts = _two_ring_7_7(r, delta)
            sc = _update_best(pts)
            if sc > best_two_ring_score:
                best_two_ring_score = sc
                best_two_ring = [r, delta]

    if best_two_ring is not None and _time_left() > 20:
        # Refine best two-ring
        try:
            def _neg_two_ring(params):
                r, delta = params
                if r <= 0:
                    r = 1e-6
                pts = _two_ring_7_7(r, delta)
                return -_score(_normalize(pts))
            
            res = scipy.optimize.minimize(
                _neg_two_ring,
                best_two_ring,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_opt = _two_ring_7_7(res.x[0], res.x[1])
            _update_best(pts_opt)
        except Exception:
            pass

    if _time_left() < 30:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 2: Explore 3-ring configurations
    # Use differential evolution for global search in 3-ring space
    # Config: (n1, n2, n3)
    three_ring_splits = [(7, 5, 2), (6, 6, 2), (5, 5, 4), (4, 5, 5)]
    
    for n1, n2, n3 in three_ring_splits:
        if _time_left() < 20:
            break
        
        def _neg_three_ring(params):
            r1, r2, r3, rot1, rot2, rot3 = params
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
            pts = _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
            return -_score(_normalize(pts))
        
        bounds = [(0.1, 1.2), (0.1, 1.2), (0.1, 1.2),
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        
        try:
            # Use a few seeds for DE
            for seed in [42, 123]:
                if _time_left() < 15:
                    break
                de_res = scipy.optimize.differential_evolution(
                    _neg_three_ring,
                    bounds=bounds,
                    seed=seed,
                    maxiter=50,
                    tol=1e-6,
                    polish=True,
                    popsize=10
                )
                r1, r2, r3, rot1, rot2, rot3 = de_res.x
                pts_de = _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
                _update_best(pts_de)
                
                # If this result is good, refine with smooth optimizer
                if best_score > 50 and _time_left() > 15:
                    _try_optimize_smooth(pts_de, maxiter_total=2000)
        except Exception:
            pass

    if _time_left() < 15:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 3: Full flexible optimization from best point found so far
    if best_points is not None:
        _try_optimize_smooth(best_points, maxiter_total=4000)

    # Phase 4: Final exact Nelder-Mead polish
    if best_points is not None and _time_left() > 5:
        try:
            res_final = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(-1, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _normalize(reg14)
    else:
        best_points = _normalize(best_points)

    return best_points

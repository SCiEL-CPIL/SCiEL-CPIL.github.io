import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
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

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        # Scale so that hull area = 1
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    Uses log-sum-exp for smooth min triangle area.
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
    signs[signs == 0] = 1.0
    
    # Gradients of cross product with respect to coordinates
    # cross = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    # d/dxi = yj - yk
    # d/dyi = xj - xk
    # d/dxj = yk - yi
    # d/dyj = -(xk - xi)
    # d/dxk = -(yj - yi)
    # d/dyk = xj - xi
    
    d_cross_di_x = points[_J, 1] - points[_K, 1]
    d_cross_di_y = points[_J, 0] - points[_K, 0]
    
    d_cross_dj_x = points[_K, 1] - points[_I, 1]
    d_cross_dj_y = -(points[_K, 0] - points[_I, 0])
    
    d_cross_dk_x = -(points[_J, 1] - points[_I, 1])
    d_cross_dk_y = points[_J, 0] - points[_I, 0]
    
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

def _refine_points(points, time_limit=10.0):
    """
    Refine a point configuration using gradient-based optimization with smooth surrogate.
    """
    start = time.time()
    deadline = start + time_limit
    
    init_norm = _normalize(points)
    x = init_norm.flatten()
    
    # Anneal beta
    n_stages = 4
    betas = [20.0, 50.0, 200.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline - 1:
            break
        
        def _obj_and_grad(x_flat, beta_val=beta):
            return _neg_smooth_score_and_grad(x_flat, beta_val)
        
        try:
            res = scipy.optimize.minimize(
                _obj_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            x = res.x
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on exact score
    if time.time() < deadline - 1:
        try:
            def _neg_exact_score(x_flat):
                return -_score(x_flat.reshape(-1, 2))
            
            res = scipy.optimize.minimize(
                _neg_exact_score,
                x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    # --- Structural Strategy: Three-Ring Symmetric Configurations ---
    # Represent 14 points as union of 3 concentric regular polygons.
    # Sizes: (7, 5, 2), (6, 6, 2), etc.
    # Parameters: r2, r3, rot2, rot3 (4 parameters)
    
    splits = [
        (7, 5, 2),
        (6, 6, 2),
        (8, 4, 2),
        (7, 4, 3),
        (5, 6, 3),
        (4, 5, 5),
        (3, 5, 6),
        (2, 6, 6),
    ]
    
    for n1, n2, n3 in splits:
        if time.time() > deadline - 5:
            break
            
        def _three_ring_score(params, n1=n1, n2=n2, n3=n3):
            r2 = params[0]
            r3 = params[1]
            rot2 = params[2]
            rot3 = params[3]
            
            if r2 <= 0.05 or r3 <= 0.05:
                return 1e6
            
            pts = _generate_three_ring_config(n1, n2, n3, 1.0, r2, r3, 0.0, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        bounds = [
            (0.1, 0.95),
            (0.05, 0.95),
            (0.0, 2 * np.pi),
            (0.0, 2 * np.pi)
        ]
        
        try:
            # Differential Evolution for global search in 4D parameter space
            res = scipy.optimize.differential_evolution(
                _three_ring_score,
                bounds,
                maxiter=30,
                popsize=8,
                seed=42,
                tol=0.01,
                polish=True
            )
            
            if res.success or res.fun < 1e5:
                best_params = res.x
                pts = _generate_three_ring_config(n1, n2, n3, 1.0, best_params[0], best_params[1], 0.0, best_params[2], best_params[3])
                sc = _update_best(pts)
                
                # If this is a good candidate, refine the full 28D coordinates
                if sc > best_score * 0.9 and time.time() < deadline - 15:
                    refined = _refine_points(pts, time_limit=10.0)
                    _update_best(refined)
                    
        except Exception:
            pass

    # --- Fallback: Standard 2-Ring searches if time remains ---
    if time.time() < deadline - 20:
        splits_2 = [(7, 7), (6, 8), (8, 6)]
        
        for n1, n2 in splits_2:
            if time.time() > deadline - 10:
                break
                
            def _two_ring_score(params, n1=n1, n2=n2):
                r2 = params[0]
                rot2 = params[1]
                if r2 <= 0.05:
                    return 1e6
                pts = _generate_three_ring_config(n1, n2, 0, 1.0, r2, 0.0, 0.0, rot2, 0.0)
                sc = _score(_normalize(pts))
                return -sc

            bounds = [
                (0.1, 0.95),
                (0.0, 2 * np.pi)
            ]
            
            try:
                res = scipy.optimize.differential_evolution(
                    _two_ring_score,
                    bounds,
                    maxiter=20,
                    popsize=6,
                    seed=42,
                    tol=0.01,
                    polish=True
                )
                if res.success or res.fun < 1e5:
                    r2, rot2 = res.x
                    pts = _generate_three_ring_config(n1, n2, 0, 1.0, r2, 0.0, 0.0, rot2, 0.0)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.9 and time.time() < deadline - 10:
                        refined = _refine_points(pts, time_limit=5.0)
                        _update_best(refined)
            except Exception:
                pass

    # Regular polygon fallback
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

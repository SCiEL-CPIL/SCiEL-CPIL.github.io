import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

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

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        # Scale so hull area = 1; ratio min_area/hull_area is scale-invariant
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 1000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                if time.time() < deadline - 1:
                    def _neg_exact_score(x_flat):
                        return -_score(x_flat.reshape(-1, 2))
                    
                    res = scipy.optimize.minimize(
                        _neg_exact_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore two-ring configurations
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2)]
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            if r2 <= 0.01:
                return -1e6
                
            pts1 = _regular_polygon(n1, r1, rot1)
            pts2 = _regular_polygon(n2, r2, rot2)
            pts = np.vstack([pts1, pts2])
            sc = _score(_normalize(pts))
            return -sc

        for r_init in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for rot_init in [0.0, np.pi / n1, np.pi / n2, 2 * np.pi / n1, np.pi / (n1 + n2)]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts1 = _regular_polygon(n1, 1.0, rot1)
                    pts2 = _regular_polygon(n2, r_ratio, rot2)
                    pts = np.vstack([pts1, pts2])
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.95 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, maxiter=1500)
                        
                except Exception:
                    pass

    # 2. Explore three-ring configurations
    three_ring_splits = [(5, 4, 5), (4, 5, 5), (5, 5, 4), (6, 4, 4), (4, 6, 4), (4, 4, 6), (7, 3, 4), (3, 7, 4)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline - 5:
            break
        
        def _three_ring_score(params):
            r2_ratio = params[0]
            r3_ratio = params[1]
            rot1 = params[2]
            rot2 = params[3]
            rot3 = params[4]
            
            r1 = 1.0
            r2 = r2_ratio * r1
            r3 = r3_ratio * r1
            
            if r2 <= 0.01 or r3 <= 0.01:
                return -1e6
                
            pts1 = _regular_polygon(n1, r1, rot1)
            pts2 = _regular_polygon(n2, r2, rot2)
            pts3 = _regular_polygon(n3, r3, rot3)
            pts = np.vstack([pts1, pts2, pts3])
            sc = _score(_normalize(pts))
            return -sc

        for r2_init in [0.4, 0.5, 0.6]:
            for r3_init in [0.2, 0.3, 0.4]:
                if time.time() > deadline - 5:
                    continue
                x0 = np.array([r2_init, r3_init, 0.0, np.pi / n2, np.pi / n3])
                try:
                    res = scipy.optimize.minimize(
                        _three_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r2_ratio, r3_ratio, rot1, rot2, rot3 = res.x
                    pts1 = _regular_polygon(n1, 1.0, rot1)
                    pts2 = _regular_polygon(n2, r2_ratio, rot2)
                    pts3 = _regular_polygon(n3, r3_ratio, rot3)
                    pts = np.vstack([pts1, pts2, pts3])
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.95 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, maxiter=1000)
                        
                except Exception:
                    pass

    # 3. Regular polygon and perturbations
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    for i in range(10):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, maxiter=1000)

    # 4. Random restarts
    for i in range(5):
        if time.time() > deadline - 10:
            break
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_smooth(init, maxiter=800)

    # 5. Coordinate Descent Polish
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        for iteration in range(3):
            if time.time() > deadline - 2:
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
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        _update_best(pts)

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

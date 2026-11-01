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

def _soft_min_triangle_area(points, beta=100.0):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    try:
        if len(np.unique(points, axis=0)) < 3:
            return 0.0
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

def _neg_soft_score(points_flat, beta=100.0):
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

def _generate_7_7_regular(r, delta):
    n = 7
    angles_outer = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    return np.vstack([points_outer, points_inner])

def _generate_7_7_free(params):
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

def _generate_5_4_5(params):
    """
    5 outer (unit circle), 4 middle (radius r1), 5 inner (radius r2).
    params: [r1, r2, theta0..4, phi0..3, psi0..4]
    Total: 2 + 5 + 4 + 5 = 16 params
    """
    r1 = params[0]
    r2 = params[1]
    theta = params[2:7]
    phi = params[7:11]
    psi = params[11:16]
    
    p_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    p_mid = np.column_stack([r1 * np.cos(phi), r1 * np.sin(phi)])
    p_inner = np.column_stack([r2 * np.cos(psi), r2 * np.sin(psi)])
    
    return np.vstack([p_outer, p_mid, p_inner])

def _neg_soft_score_5_4_5(params, beta=100.0):
    try:
        points = _generate_5_4_5(params)
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_score_5_4_5(params):
    try:
        points = _generate_5_4_5(params)
        return -_score(points)
    except Exception:
        return 0.0

def _generate_6_5_3(params):
    """
    6 outer, 5 middle, 3 inner.
    params: [r1, r2, theta0..5, phi0..4, psi0..2]
    Total: 2 + 6 + 5 + 3 = 16 params
    """
    r1 = params[0]
    r2 = params[1]
    theta = params[2:8]
    phi = params[8:13]
    psi = params[13:16]
    
    p_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    p_mid = np.column_stack([r1 * np.cos(phi), r1 * np.sin(phi)])
    p_inner = np.column_stack([r2 * np.cos(psi), r2 * np.sin(psi)])
    
    return np.vstack([p_outer, p_mid, p_inner])

def _neg_soft_score_6_5_3(params, beta=100.0):
    try:
        points = _generate_6_5_3(params)
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_score_6_5_3(params):
    try:
        points = _generate_6_5_3(params)
        return -_score(points)
    except Exception:
        return 0.0

def _generate_7_5_2(params):
    """
    7 outer, 5 middle, 2 inner.
    params: [r1, r2, theta0..6, phi0..4, psi0..1]
    Total: 2 + 7 + 5 + 2 = 16 params
    """
    r1 = params[0]
    r2 = params[1]
    theta = params[2:9]
    phi = params[9:14]
    psi = params[14:16]
    
    p_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    p_mid = np.column_stack([r1 * np.cos(phi), r1 * np.sin(phi)])
    p_inner = np.column_stack([r2 * np.cos(psi), r2 * np.sin(psi)])
    
    return np.vstack([p_outer, p_mid, p_inner])

def _neg_soft_score_7_5_2(params, beta=100.0):
    try:
        points = _generate_7_5_2(params)
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_score_7_5_2(params):
    try:
        points = _generate_7_5_2(params)
        return -_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Phase 1: Coarse Search in 7+7 Regular Parameter Space
    def _neg_soft_score_7_7_reg(params, beta=50.0):
        r, delta = params
        if r < 0.01: r = 0.01
        if r > 1.0: r = 1.0
        pts = _generate_7_7_regular(r, delta)
        ha = _hull_area(pts)
        if ha < 1e-12: return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    
    best_reg_params = np.array([0.5, 0.0])
    
    for r_init in [0.3, 0.5, 0.7, 0.9]:
        for d_init in [0.0, np.pi/7, 2*np.pi/7, 3*np.pi/7]:
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

    # Phase 2: Fine Search in 7+7 Free Angles
    r_best, d_best = best_reg_params
    theta_init = np.linspace(0, 2*np.pi, 7, endpoint=False)
    phi_init = np.linspace(d_best, d_best + 2*np.pi, 7, endpoint=False)
    params_init = np.concatenate(([r_best], theta_init, phi_init))
    
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

    # Phase 3: Explore 5+4+5 structure
    if _time_left() > 30:
        # Initialize 5+4+5 from best points
        # Try to fit: 5 outermost, 4 middle, 5 innermost
        if best_points is not None:
            # Sort by distance from centroid
            centroid = best_points.mean(axis=0)
            dists = np.linalg.norm(best_points - centroid, axis=1)
            order = np.argsort(dists)
            # Innermost 5, middle 4, outermost 5
            inner_idx = order[:5]
            mid_idx = order[5:9]
            outer_idx = order[9:14]
            
            # Get radii and angles
            p_inner = best_points[inner_idx]
            p_mid = best_points[mid_idx]
            p_outer = best_points[outer_idx]
            
            r2_init = np.linalg.norm(p_inner - centroid, axis=1).mean()
            r1_init = np.linalg.norm(p_mid - centroid, axis=1).mean()
            
            psi_init = np.arctan2(p_inner[:, 1] - centroid[1], p_inner[:, 0] - centroid[0])
            phi_init_545 = np.arctan2(p_mid[:, 1] - centroid[1], p_mid[:, 0] - centroid[0])
            theta_init_545 = np.arctan2(p_outer[:, 1] - centroid[1], p_outer[:, 0] - centroid[0])
            
            params_545 = np.concatenate(([r1_init, r2_init], theta_init_545, phi_init_545, psi_init))
            
            for beta in [100.0, 300.0]:
                if _time_left() < 10: break
                try:
                    res = scipy.optimize.minimize(
                        lambda x, b=beta: _neg_soft_score_5_4_5(x, b),
                        params_545,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    new_pts = _generate_5_4_5(res.x)
                    sc_new = _score(new_pts)
                    if sc_new > best_score:
                        _update_best(new_pts, sc_new)
                        params_545 = res.x
                except Exception:
                    pass
            
            if _time_left() > 10:
                try:
                    res_nm = scipy.optimize.minimize(
                        _neg_score_5_4_5,
                        params_545,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    new_pts = _generate_5_4_5(res_nm.x)
                    sc_new = _score(new_pts)
                    if sc_new > best_score:
                        _update_best(new_pts, sc_new)
                except Exception:
                    pass

    # Phase 4: Explore 6+5+3 structure
    if _time_left() > 30 and best_points is not None:
        centroid = best_points.mean(axis=0)
        dists = np.linalg.norm(best_points - centroid, axis=1)
        order = np.argsort(dists)
        inner_idx = order[:3]
        mid_idx = order[3:8]
        outer_idx = order[8:14]
        
        p_inner = best_points[inner_idx]
        p_mid = best_points[mid_idx]
        p_outer = best_points[outer_idx]
        
        r2_init = np.linalg.norm(p_inner - centroid, axis=1).mean()
        r1_init = np.linalg.norm(p_mid - centroid, axis=1).mean()
        
        psi_init = np.arctan2(p_inner[:, 1] - centroid[1], p_inner[:, 0] - centroid[0])
        phi_init_653 = np.arctan2(p_mid[:, 1] - centroid[1], p_mid[:, 0] - centroid[0])
        theta_init_653 = np.arctan2(p_outer[:, 1] - centroid[1], p_outer[:, 0] - centroid[0])
        
        params_653 = np.concatenate(([r1_init, r2_init], theta_init_653, phi_init_653, psi_init))
        
        for beta in [100.0, 300.0]:
            if _time_left() < 10: break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score_6_5_3(x, b),
                    params_653,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = _generate_6_5_3(res.x)
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
                    params_653 = res.x
            except Exception:
                pass
        
        if _time_left() > 10:
            try:
                res_nm = scipy.optimize.minimize(
                    _neg_score_6_5_3,
                    params_653,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_pts = _generate_6_5_3(res_nm.x)
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    # Phase 5: Explore 7+5+2 structure
    if _time_left() > 30 and best_points is not None:
        centroid = best_points.mean(axis=0)
        dists = np.linalg.norm(best_points - centroid, axis=1)
        order = np.argsort(dists)
        inner_idx = order[:2]
        mid_idx = order[2:7]
        outer_idx = order[7:14]
        
        p_inner = best_points[inner_idx]
        p_mid = best_points[mid_idx]
        p_outer = best_points[outer_idx]
        
        r2_init = np.linalg.norm(p_inner - centroid, axis=1).mean()
        r1_init = np.linalg.norm(p_mid - centroid, axis=1).mean()
        
        psi_init = np.arctan2(p_inner[:, 1] - centroid[1], p_inner[:, 0] - centroid[0])
        phi_init_752 = np.arctan2(p_mid[:, 1] - centroid[1], p_mid[:, 0] - centroid[0])
        theta_init_752 = np.arctan2(p_outer[:, 1] - centroid[1], p_outer[:, 0] - centroid[0])
        
        params_752 = np.concatenate(([r1_init, r2_init], theta_init_752, phi_init_752, psi_init))
        
        for beta in [100.0, 300.0]:
            if _time_left() < 10: break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score_7_5_2(x, b),
                    params_752,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_pts = _generate_7_5_2(res.x)
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
                    params_752 = res.x
            except Exception:
                pass
        
        if _time_left() > 10:
            try:
                res_nm = scipy.optimize.minimize(
                    _neg_score_7_5_2,
                    params_752,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_pts = _generate_7_5_2(res_nm.x)
                sc_new = _score(new_pts)
                if sc_new > best_score:
                    _update_best(new_pts, sc_new)
            except Exception:
                pass

    # Phase 6: Full Space Optimization (28 dims)
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

    # Phase 7: Random Restarts / Perturbations
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

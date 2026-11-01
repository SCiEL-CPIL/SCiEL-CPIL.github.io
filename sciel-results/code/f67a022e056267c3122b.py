import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
_N_TRI = len(_TRI)

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
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_heptagons(params):
    """
    Two concentric heptagons parameterized by:
    params = [r_outer, r_inner, theta_outer, theta_inner, cx, cy, sx, sy]
    where sx, sy are small per-point radial perturbations (7 each, but we use
    a low-frequency Fourier mode to keep it smooth).
    
    Actually, let's use a cleaner 8-param representation:
    params = [R, r, phi_R, phi_r, dx, dy, alpha, beta]
    - R: outer radius
    - r: inner radius  
    - phi_R: outer rotation
    - phi_r: inner rotation
    - dx, dy: center offset
    - alpha, beta: shape distortion parameters (make it slightly non-circular)
    """
    R, r, phi_R, phi_r, dx, dy, alpha, beta = params
    n_h = 7
    angles_R = np.linspace(0, 2 * np.pi, n_h, endpoint=False) + phi_R
    angles_r = np.linspace(0, 2 * np.pi, n_h, endpoint=False) + phi_r
    
    # Slight elliptical distortion
    rR = R * (1.0 + alpha * 0.1 * np.cos(2 * angles_R))
    rr = r * (1.0 + beta * 0.1 * np.cos(2 * angles_r))
    
    outer = np.column_stack([rR * np.cos(angles_R), rR * np.sin(angles_R)])
    inner = np.column_stack([rr * np.cos(angles_r), rr * np.sin(angles_r)])
    
    pts = np.vstack([outer, inner])
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts

def _two_heptagons_grad(params):
    """Compute points and their Jacobian w.r.t. params."""
    R, r, phi_R, phi_r, dx, dy, alpha, beta = params
    n_h = 7
    angles_R = np.linspace(0, 2 * np.pi, n_h, endpoint=False) + phi_R
    angles_r = np.linspace(0, 2 * np.pi, n_h, endpoint=False) + phi_r
    
    cos2R = np.cos(2 * angles_R)
    sin2R = np.sin(2 * angles_R)
    cos2r = np.cos(2 * angles_r)
    sin2r = np.sin(2 * angles_r)
    
    rR = R * (1.0 + alpha * 0.1 * cos2R)
    rr = r * (1.0 + beta * 0.1 * cos2r)
    
    outer = np.column_stack([rR * np.cos(angles_R), rR * np.sin(angles_R)])
    inner = np.column_stack([rr * np.cos(angles_r), rr * np.sin(angles_r)])
    
    pts = np.vstack([outer, inner])
    pts[:, 0] += dx
    pts[:, 1] += dy
    
    # Jacobian: 14x2 points x 8 params -> shape (14, 2, 8)
    J = np.zeros((14, 2, 8))
    
    # d outer/dR
    J[:7, 0, 0] = (1.0 + alpha * 0.1 * cos2R) * np.cos(angles_R)
    J[:7, 1, 0] = (1.0 + alpha * 0.1 * cos2R) * np.sin(angles_R)
    
    # d outer/dr
    J[7:, 0, 1] = (1.0 + beta * 0.1 * cos2r) * np.cos(angles_r)
    J[7:, 1, 1] = (1.0 + beta * 0.1 * cos2r) * np.sin(angles_r)
    
    # d outer/dphi_R
    J[:7, 0, 2] = -rR * np.sin(angles_R)
    J[:7, 1, 2] = rR * np.cos(angles_R)
    
    # d inner/dphi_r
    J[7:, 0, 3] = -rr * np.sin(angles_r)
    J[7:, 1, 3] = rr * np.cos(angles_r)
    
    # d/dx, d/dy
    J[:, 0, 4] = 1.0
    J[:, 1, 5] = 1.0
    
    # d outer/dalpha
    J[:7, 0, 6] = 0.1 * R * cos2R * np.cos(angles_R)
    J[:7, 1, 6] = 0.1 * R * cos2R * np.sin(angles_R)
    
    # d inner/dbeta
    J[7:, 0, 7] = 0.1 * r * cos2r * np.cos(angles_r)
    J[7:, 1, 7] = 0.1 * r * cos2r * np.sin(angles_r)
    
    return pts, J

def _soft_min_score(params, tau=0.05):
    """
    Smooth approximation of min triangle area / hull area.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts, J = _two_heptagons_grad(params)
    
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    soft_min = max_a + tau * np.log(np.sum(exp_terms))
    
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros(8)
    
    value = soft_min / ha
    
    # Gradient via finite differences of the smooth objective
    # (analytical gradient is complex; FD is acceptable for 8 params)
    grad = np.zeros(8)
    h = 1e-6
    for i in range(8):
        params_p = params.copy()
        params_p[i] += h
        val_p = _soft_min_score_value(params_p, tau)
        grad[i] = (val_p - value) / h
    
    return value, -grad  # negate for minimization

def _soft_min_score_value(params, tau=0.05):
    """Just the value, no gradient."""
    pts, _ = _two_heptagons_grad(params)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    soft_min = max_a + tau * np.log(np.sum(exp_terms))
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    return soft_min / ha

def _exact_maxmin_obj_grad(params, tau=0.01):
    """
    Maximize min triangle area / hull area.
    Returns (objective, gradient) where objective is -min_area/hull_area
    and gradient is d(-min_area/hull_area)/dparams.
    Uses the triangle with minimum area for exact gradient.
    """
    pts, J = _two_heptagons_grad(params)
    
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros(8)
    
    obj = -min_area / ha
    
    # Gradient of min_area w.r.t. params
    # For the critical triangle (i,j,k): area = 0.5 * |cross|
    # d(area)/d(params) = 0.5 * sign(cross) * d(cross)/d(params)
    i, j, k = _TRI[min_idx]
    
    # cross = (pj_i - pi_i) * (pk_j - pi_j) - (pj_j - pi_j) * (pk_i - pi_i)
    # d(cross)/d(pts_m) for each point m
    # Let's compute d(cross)/d(pts) analytically
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i = -(y_k - y_i)
    # d/dy_i = (x_k - x_i)
    # d/dx_j = (y_k - y_i)
    # d/dy_j = -(x_k - x_i)
    # d/dx_k = -(y_j - y_i)
    # d/dy_k = (x_j - x_i)
    
    sign_c = np.sign(cross[min_idx])
    if sign_c == 0:
        sign_c = 1.0
    
    d_cross_d_pts = np.zeros((14, 2))
    xi, yi = pts[i]
    xj, yj = pts[j]
    xk, yk = pts[k]
    
    d_cross_d_pts[i, 0] = -(yk - yi)
    d_cross_d_pts[i, 1] = (xk - xi)
    d_cross_d_pts[j, 0] = (yk - yi)
    d_cross_d_pts[j, 1] = -(xk - xi)
    d_cross_d_pts[k, 0] = -(yj - yi)
    d_cross_d_pts[k, 1] = (xj - xi)
    
    d_area_d_pts = 0.5 * sign_c * d_cross_d_pts
    
    # d(area)/d(params) = sum_m d_area_d_pts[m] @ J[m]
    d_area_d_params = np.einsum('md,mdp->p', d_area_d_pts, J)
    
    # For hull area, use finite difference (complex to do analytically)
    h = 1e-6
    d_ha_d_params = np.zeros(8)
    for p in range(8):
        params_p = params.copy()
        params_p[p] += h
        pts_p, _ = _two_heptagons_grad(params_p)
        ha_p = _hull_area(pts_p)
        d_ha_d_params[p] = (ha_p - ha) / h
    
    # d(obj)/d(params) = -d(min_area/hull_area)/d(params)
    # = -(d_area * ha - min_area * d_ha) / ha^2
    d_obj_d_params = -(d_area_d_params * ha - min_area * d_ha_d_params) / (ha * ha)
    
    return obj, d_obj_d_params

def _build_structured_starts():
    starts = []
    # Regular polygon baseline
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3, np.pi / 7]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    # Two-heptagon configurations
    for R in [1.0, 1.05, 1.1]:
        for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for phi_R in [0.0, np.pi/7, 2*np.pi/7]:
                for phi_r in [0.0, np.pi/7, 2*np.pi/7, np.pi/14]:
                    params = np.array([R, r, phi_R, phi_r, 0.0, 0.0, 0.0, 0.0])
                    starts.append(_two_heptagons(params))
    
    # Elliptical hulls
    for a in [0.9, 1.0, 1.1]:
        for b in [0.7, 0.8, 0.9, 1.0]:
            for rot in [0.0, np.pi/6, np.pi/4, np.pi/3]:
                theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + rot
                x = a * np.cos(theta)
                y = b * np.sin(theta)
                pts = np.column_stack([x, y])
                c, s = np.cos(rot), np.sin(rot)
                R_mat = np.array([[c, -s], [s, c]])
                starts.append(pts @ R_mat.T)
    
    return starts

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score, best_pts = 0.0, _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    # Phase 1: Differential evolution on two-heptagon parameters
    if left() > 40.0:
        try:
            bounds = [(0.5, 1.5), (0.1, 0.9), (-np.pi, np.pi), (-np.pi, np.pi),
                      (-0.3, 0.3), (-0.3, 0.3), (-1.0, 1.0), (-1.0, 1.0)]
            
            def de_obj(params):
                return _soft_min_score_value(params, tau=0.05)
            
            de_res = scipy.optimize.differential_evolution(
                de_obj, bounds, maxiter=100, popsize=15, tol=1e-8,
                seed=42, polish=True, update_interval=10
            )
            best_params = de_res.x
            best_pts_de = _two_heptagons(best_params)
            best_pts_de = _normalize(best_pts_de)
            sc_de = _score(best_pts_de)
            if sc_de > best_score:
                best_score, best_pts = sc_de, best_pts_de.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B with exact gradient on two-heptagon params
    if left() > 25.0:
        # Find good initial params from best_pts
        # Try to fit two-heptagons to best_pts
        best_params = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Also try from DE result if available
        if 'de_res' in dir():
            best_params = de_res.x
        
        try:
            def lbfgs_obj(params):
                obj, grad = _exact_maxmin_obj_grad(params, tau=0.01)
                return obj, grad
            
            res = scipy.optimize.minimize(
                lbfgs_obj, best_params, method='L-BFGS-B',
                jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            opt_params = res.x
            opt_pts = _two_heptagons(opt_params)
            opt_pts = _normalize(opt_pts)
            sc = _score(opt_pts)
            if sc > best_score:
                best_score, best_pts = sc, opt_pts.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead on top free-space candidates
    top_candidates = [pts for _, pts in scored[:3]]
    for pts in top_candidates:
        if left() <= 15.0:
            break
        try:
            res = scipy.optimize.minimize(
                _neg_score, pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 15000, 'fatol': 1e-13, 'xatol': 1e-10, 'adaptive': True, 'maxfev': 20000}
            )
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Phase 4: Final polish with tight Nelder-Mead
    if left() > 8.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 8000, 'fatol': 1e-15, 'xatol': 1e-12, 'adaptive': True, 'maxfev': 15000}
            )
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    return _normalize(best_pts)

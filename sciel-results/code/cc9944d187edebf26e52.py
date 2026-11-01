import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0  # wall-clock guard (evaluator kills at 150s)

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
_NTRI = _TRI.shape[0]

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

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    """Score for two-nested-heptagons parameterization."""
    try:
        r_inner, phi = params
        if r_inner < 0:
            r_inner = 0.0
        points = _two_heptagons(r_inner, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_heptagons(params):
    return -_score_two_heptagons(params)

def _exact_maxmin_obj_grad(points):
    """
    Returns (value, grad) for f(points) = -min_triangle_area(points) / hull_area(points).
    Uses the EXACT minimum (no LSE). The subgradient of min_j a_j is a unit vector on
    the set of minimizers, which is valid for maximizing the minimum.
    CRITICAL: Must multiply by sign(cross) because area = 0.5 * |cross|.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    # Sign of cross product for absolute value gradient
    signs = np.sign(cross)
    # Where cross is 0, sign is 0, but area is 0 so it won't be in mask unless all are 0
    # If all are 0, gradient is 0 anyway. Set to 1 for safety.
    signs[signs == 0] = 1.0

    # Gradient of cross product w.r.t points
    # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # d/dxi = -(yk-yi) - (-(yj-yi)) = yj - yk
    # d/dyi = (xj-xi) - (-(xk-xi)) = xj - xk
    # d/dxj = (yk-yi)
    # d/dyj = -(xk-xi)
    # d/dxk = -(yj-yi)
    # d/dyk = (xj-xi)
    
    g_a_x = v1[:, 1] - v2[:, 1]  # yj - yk
    g_a_y = v2[:, 0] - v1[:, 0]  # xk - xj  (Wait: d/dyi = xj - xk. Let's re-derive)
    
    # Let's re-verify gradients carefully.
    # C = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # dC/dxi = -(yk-yi) + (yj-yi) = yj - yk
    # dC/dyi = (xj-xi) - (xk-xi) = xj - xk
    # dC/dxj = (yk-yi)
    # dC/dyj = -(xk-xi)
    # dC/dxk = -(yj-yi)
    # dC/dyk = (xj-xi)
    
    # v1 = pj - pi, v2 = pk - pi
    # v1[:,1] = yj-yi, v2[:,1] = yk-yi
    # g_a_x (dC/dxi) = v1[:,1] - v2[:,1]
    # g_a_y (dC/dyi) = v1[:,0] - v2[:,0]
    
    # g_b_x (dC/dxj) = v2[:,1]
    # g_b_y (dC/dyj) = -v2[:,0]
    
    # g_c_x (dC/dxk) = -v1[:,1]
    # g_c_y (dC/dyk) = v1[:,0]
    
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v1[:, 0] - v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    # Gradient of area = 0.5 * |cross|
    # d(area)/d(points) = 0.5 * sign(cross) * d(cross)/d(points)
    wh = 0.5 * w * signs

    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), wh * g_a_x)
    np.add.at(grad, (_I, 1), wh * g_a_y)
    np.add.at(grad, (_J, 0), wh * g_b_x)
    np.add.at(grad, (_J, 1), wh * g_b_y)
    np.add.at(grad, (_K, 0), wh * g_c_x)
    np.add.at(grad, (_K, 1), wh * g_c_y)

    val = -m / ha
    grad = -grad / ha
    return val, grad

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for soft-min objective."""
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros((N, 2))
    except Exception:
        return 0.0, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp  # softmax weights
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t. areas
    g_areas = -probs
    
    # Gradient of areas w.r.t. points
    # Same as before, but areas are positive so no sign flip needed for soft_min(areas)
    # However, soft_min is on areas. The gradient of area wrt points still needs the sign of cross?
    # No, soft_min(A) is smooth. d(soft_min)/d(points) = sum(p_j * d(A_j)/d(points)).
    # d(A_j)/d(points) = 0.5 * sign(cross_j) * d(cross_j)/d(points).
    # So we MUST include the sign here too!
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v1[:, 0] - v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    # g_areas is d(soft_min)/d(area). 
    # d(soft_min)/d(points) = g_areas * d(area)/d(points)
    # d(area)/d(points) = 0.5 * sign * d(cross)/d(points)
    
    coeff = g_areas * 0.5 * signs
    
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), coeff * g_a_x)
    np.add.at(grad, (_I, 1), coeff * g_a_y)
    np.add.at(grad, (_J, 0), coeff * g_b_x)
    np.add.at(grad, (_J, 1), coeff * g_b_y)
    np.add.at(grad, (_K, 0), coeff * g_c_x)
    np.add.at(grad, (_K, 1), coeff * g_c_y)
    
    val = -soft_min_val / ha
    grad = -grad / ha
    return val, grad

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    def try_update(points):
        nonlocal best_score, best_pts
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_pts = points.copy()
        return sc
    
    # ---- Phase 0: Two-nested-heptagons 2D global search (fast, exploits symmetry) ----
    if left() > 5.0:
        # Dense grid over (r_inner, phi)
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        # Golden-section refinement on the best region
        if left() > 5.0:
            # Find best (r, phi) from grid
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            # 1D golden-section on r for fixed phi
            gr = 0.6180339887498949
            a_r, b_r = max(0.0, best_r - 0.1), min(1.0, best_r + 0.1)
            c_r = b_r - gr * (b_r - a_r)
            d_r = a_r + gr * (b_r - a_r)
            fc = _score_two_heptagons([c_r, best_phi])
            fd = _score_two_heptagons([d_r, best_phi])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_r = c_r
                    c_r = d_r
                    fc = fd
                    d_r = b_r - gr * (b_r - a_r)
                    fd = _score_two_heptagons([d_r, best_phi])
                else:
                    b_r = d_r
                    d_r = c_r
                    fd = fc
                    c_r = a_r + gr * (b_r - a_r)
                    fc = _score_two_heptagons([c_r, best_phi])
            best_r = (a_r + b_r) / 2
            
            # 1D golden-section on phi for fixed r
            a_p, b_p = 0.0, np.pi / 7
            c_p = b_p - gr * (b_p - a_p)
            d_p = a_p + gr * (b_p - a_p)
            fc = _score_two_heptagons([best_r, c_p])
            fd = _score_two_heptagons([best_r, d_p])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_p = c_p
                    c_p = d_p
                    fc = fd
                    d_p = b_p - gr * (b_p - a_p)
                    fd = _score_two_heptagons([best_r, d_p])
                else:
                    b_p = d_p
                    d_p = c_p
                    fd = fc
                    c_p = a_p + gr * (b_p - a_p)
                    fc = _score_two_heptagons([best_r, c_p])
            best_phi = (a_p + b_p) / 2
            
            pts = _two_heptagons(best_r, best_phi)
            try_update(pts)
    
    # ---- Phase 1: Differential evolution on 4-ring parameters (global search) ----
    if left() > 60.0:
        def _ring_configuration(params):
            r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
            n1 = int(round(n1))
            n2 = int(round(n2))
            n3 = int(round(n3))
            n4 = int(round(n4))
            
            total = n1 + n2 + n3 + n4
            while total > N:
                if n4 >= 1:
                    n4 -= 1
                elif n3 >= 1:
                    n3 -= 1
                elif n2 >= 1:
                    n2 -= 1
                else:
                    n1 -= 1
                total -= 1
            while total < N:
                n4 += 1
                total += 1
            
            points = []
            
            if n1 > 0:
                angles = th1 + np.linspace(0, 2 * np.pi, n1, endpoint=False)
                points.append(np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles))))
            
            if n2 > 0:
                angles = th2 + np.linspace(0, 2 * np.pi, n2, endpoint=False)
                points.append(np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles))))
            
            if n3 > 0:
                angles = th3 + np.linspace(0, 2 * np.pi, n3, endpoint=False)
                points.append(np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles))))
            
            if n4 > 0:
                angles = th4 + np.linspace(0, 2 * np.pi, n4, endpoint=False)
                points.append(np.column_stack((r4 * np.cos(angles), r4 * np.sin(angles))))
            
            if not points:
                return np.zeros((N, 2))
            
            result = np.vstack(points)
            
            if len(result) < N:
                extra = np.zeros((N - len(result), 2))
                result = np.vstack([result, extra])
            elif len(result) > N:
                result = result[:N]
            
            return result

        def _neg_ring_score(params):
            try:
                points = _ring_configuration(params)
                return -_score(points)
            except Exception:
                return 0.0

        bounds = [
            (0.3, 1.5),   # r1 (outermost)
            (0.1, 1.0),   # r2
            (0.0, 0.7),   # r3
            (0.0, 0.4),   # r4 (innermost)
            (0, 2 * np.pi),  # th1
            (0, 2 * np.pi),  # th2
            (0, 2 * np.pi),  # th3
            (0, 2 * np.pi),  # th4
            (1, 14),      # n1
            (1, 14),      # n2
            (1, 14),      # n3
            (1, 14),      # n4
        ]
        
        rng = np.random.default_rng(42)
        for restart in range(3):
            if left() <= 40.0:
                break
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_ring_score,
                    bounds,
                    seed=rng.integers(0, 2**31),
                    maxiter=60,
                    tol=1e-5,
                    popsize=10,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    polish=True,
                    init='latinhypercube'
                )
                if res.success or res.fun < 0:
                    points = _ring_configuration(res.x)
                    try_update(points)
            except Exception:
                pass
    
    # ---- Phase 2: Structured starts + gradient-based local refinement ----
    if left() > 25.0:
        # Build diverse starting points
        starts = [best_pts.copy()]
        
        # Regular polygon
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        # Two rings 7+7
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
        ))))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6 * np.cos(a2), 0.6 * np.sin(a2))),
            np.column_stack((0.25 * np.cos(a3), 0.25 * np.sin(a3)))
        ))))
        
        # Jittered versions of best
        rng2 = np.random.default_rng(123)
        for _ in range(4):
            if left() <= 15.0:
                break
            jittered = best_pts + rng2.normal(0, 0.08, (N, 2))
            starts.append(_normalize(jittered))
        
        def try_opt(x0, method, maxiter, fatol, xatol, use_grad=False):
            nonlocal best_score, best_pts
            if left() <= 4.0:
                return
            try:
                if use_grad:
                    def obj_and_grad(flat):
                        val, grad = _exact_maxmin_obj_grad(flat.reshape(-1, 2))
                        return val, grad.flatten()
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                            'maxfev': 10000, 'maxls': 40}
                    res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                                  jac=True, options=opts)
                else:
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 15000}
                    if method == 'Nelder-Mead':
                        opts['adaptive'] = True
                    res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
        
        # Local refinement on each start
        for start in starts:
            if left() <= 10.0:
                break
            try_opt(start, 'L-BFGS-B', 2000, 1e-12, 1e-11, use_grad=True)
            if left() <= 8.0:
                break
            try_opt(start, 'Nelder-Mead', 3000, 1e-12, 1e-9)
    
    # ---- Phase 3: Soft-min annealing for smooth global navigation ----
    if left() > 20.0:
        tau_vals = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        for tau in tau_vals:
            if left() <= 5.0:
                break
            try:
                def obj_and_grad(flat):
                    val, grad = _soft_min_score_grad(flat, tau)
                    return val, grad.flatten()
                res = scipy.optimize.minimize(obj_and_grad, best_pts.flatten(), method='L-BFGS-B',
                                              jac=True, options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-11})
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
    
    # ---- Phase 4: Fine polish on the best configuration ----
    if left() > 15.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-13, 1e-12, use_grad=True)
    if left() > 12.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-14, 1e-11)
    if left() > 10.0:
        try_opt(best_pts, 'L-BFGS-B', 2000, 1e-14, 1e-13, use_grad=True)
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 3000, 1e-15, 1e-13)
    if left() > 6.0:
        try_opt(best_pts, 'L-BFGS-B', 1500, 1e-15, 1e-14, use_grad=True)
    
    return _normalize(best_pts)

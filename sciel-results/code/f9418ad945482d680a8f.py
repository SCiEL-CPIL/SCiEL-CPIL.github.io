import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 142.0  # wall-clock guard (evaluator kills at 150s)

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

def _two_ring_config(n, frac=0.5, phase=0.5):
    n_outer = n - 7
    n_inner = 7
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, frac, phase * 2 * np.pi / n_inner)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.5, 0.25, 0.75, 1/7, 2/7]:
            starts.append(_two_ring_config(N, frac, phase))
    # Ellipse configurations with varied aspect ratios and rotations
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95)]:
        for rot in [0.0, np.pi / 6, np.pi / 4]:
            starts.append(_ellipse_config(N, a, b, rot))
    return starts

def _lse_min_area_and_grad(points, tau=0.01):
    """
    Compute LSE approximation of min triangle area AND its gradient.
    
    soft_min(areas) = -tau * log(sum(exp(-areas/tau)))
    d(soft_min)/d(area_j) = exp(-area_j/tau) / sum(exp(-area_k/tau)) = w_j
    
    For cross_j = v1_j × v2_j where v1 = pj-pi, v2 = pk-pi:
    d(cross_j)/d(pi) = v2_j × v1_j = -cross_j (but as a vector: (v2_y, -v2_x) is wrong)
    
    Actually: cross = (pj_x-pi_x)*(pk_y-pi_y) - (pj_y-pi_y)*(pk_x-pi_x)
    d(cross)/d(pi_x) = -(pk_y-pi_y) + (pj_y-pi_y) = v1_y - v2_y
    d(cross)/d(pi_y) = (pk_x-pi_x) - (pj_x-pi_x) = v2_x - v1_x
    So d(cross)/d(pi) = (v1_y - v2_y, v2_x - v1_x) = v1×v2 direction... 
    
    Let me recompute:
    cross = (pj_x-pi_x)*(pk_y-pi_y) - (pj_y-pi_y)*(pk_x-pi_x)
    Let a = pj_x-pi_x, b = pj_y-pi_y, c = pk_x-pi_x, d = pk_y-pi_y
    cross = a*d - b*c
    
    d(cross)/d(pi_x) = -d + b = b - d = v1_y - v2_y
    d(cross)/d(pi_y) = -(-c) + a... wait
    d(cross)/d(pi_y) = a*0 - 1*c... no.
    cross = a*d - b*c
    d(cross)/d(pi_y): a depends on pi_y? No, a = pj_x-pi_x doesn't depend on pi_y.
    b = pj_y-pi_y, so db/d(pi_y) = -1
    d(cross)/d(pi_y) = a*0 - (-1)*c = c = v2_x... 
    
    Hmm, let me be more careful:
    cross = (pj_x-pi_x)*(pk_y-pi_y) - (pj_y-pi_y)*(pk_x-pi_x)
    
    d(cross)/d(pi_x) = [-(pk_y-pi_y)] + [-(pj_y-pi_y)]*(-1) = -(pk_y-pi_y) + (pj_y-pi_y) = v1_y - v2_y
    d(cross)/d(pi_y) = [(pj_x-pi_x)]*(-1) - [(pj_y-pi_y)] = -(pj_x-pi_x) - (pj_y-pi_y)... 
    
    Wait, I need to be more careful with the product rule.
    Let f = (pj_x-pi_x)*(pk_y-pi_y) and g = (pj_y-pi_y)*(pk_x-pi_x)
    cross = f - g
    
    df/d(pi_x) = -(pk_y-pi_y) = -v2_y
    df/d(pi_y) = (pj_x-pi_x)*(-1) = -v1_x
    dg/d(pi_x) = -(pj_y-pi_y) = -v1_y
    dg/d(pi_y) = (pj_y-pi_y)*(-1) = -v1_y... no wait
    dg/d(pi_y) = [d(pj_y-pi_y)/d(pi_y)]*(pk_x-pi_x) + (pj_y-pi_y)*[d(pk_x-pi_x)/d(pi_y)]
    = (-1)*(pk_x-pi_x) + 0 = -v2_x
    
    So d(cross)/d(pi_x) = -v2_y - (-v1_y) = v1_y - v2_y
    d(cross)/d(pi_y) = -v1_x - (-v2_x) = v2_x - v1_x
    
    d(cross)/d(pj_x) = (pk_y-pi_y) = v2_y
    d(cross)/d(pj_y) = -(pk_x-pi_x) = -v2_x
    
    d(cross)/d(pk_x) = -(pj_y-pi_y) = -v1_y
    d(cross)/d(pk_y) = (pj_x-pi_x) = v1_x
    
    So:
    d(cross)/d(pi) = (v1_y - v2_y, v2_x - v1_x)
    d(cross)/d(pj) = (v2_y, -v2_x)
    d(cross)/d(pk) = (-v1_y, v1_x)
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # LSE min: soft_min(areas) = -tau * log(sum(exp(-areas/tau)))
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = -tau * np.log(sum_exp) + max_a  # = max_a - tau*log(sum_exp)
    
    # Weights for gradient
    w = exp_terms / sum_exp
    
    # d(soft_min)/d(area_j) = w_j
    # area_j = 0.5 * |cross_j|
    # Assuming convex config (positive cross), area_j = 0.5 * cross_j
    # d(area_j)/d(points) = 0.5 * d(cross_j)/d(points)
    
    grad = np.zeros((N, 2))
    
    # d(cross)/d(pi) = (v1_y - v2_y, v2_x - v1_x)
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    
    # d(cross)/d(pj) = (v2_y, -v2_x)
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    
    # d(cross)/d(pk) = (-v1_y, v1_x)
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    # Weighted sums: d(soft_min)/d(p_i) = sum_j w_j * 0.5 * d(cross_j)/d(p_i)
    w_half = 0.5 * w
    
    # Accumulate into grad
    np.add.at(grad, (_I, 0), w_half * g_a_x)
    np.add.at(grad, (_I, 1), w_half * g_a_y)
    np.add.at(grad, (_J, 0), w_half * g_b_x)
    np.add.at(grad, (_J, 1), w_half * g_b_y)
    np.add.at(grad, (_K, 0), w_half * g_c_x)
    np.add.at(grad, (_K, 1), w_half * g_c_y)
    
    return lse_val, grad

def _smooth_neg_score_and_grad(flat, tau=0.01):
    """
    Smooth objective for L-BFGS-B with analytical gradient.
    Returns (value, gradient) of -min_area_lse / hull_area.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros(2 * N)
    
    lse_val, lse_grad = _lse_min_area_and_grad(pts, tau)
    
    # Objective: f = -lse_val / ha
    # df/dp = -(lse_grad * ha) / ha^2 = -lse_grad / ha (ignoring d(ha)/dp)
    
    val = -lse_val / ha
    grad = -lse_grad / ha
    
    return val, grad.flatten()

def _smooth_neg_score_annealed(flat, tau_schedule):
    """
    Smooth objective with annealed tau for L-BFGS-B.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    
    current_iter = _smooth_neg_score_annealed.current_iter
    tau = tau_schedule[-1][0]
    for t, threshold in tau_schedule:
        if current_iter < threshold:
            tau = t
            break
    
    min_a_lse = _lse_min_area_and_grad(pts, tau)[0]
    return -min_a_lse / ha

_smooth_neg_score_annealed.current_iter = 0

def _circle_min_area_and_grad(angles, tau=0.01):
    """
    For points on unit circle at given angles, compute LSE min triangle area and gradient.
    Triangle area for points at angles a,b,c on unit circle:
    area = 0.5 * |sin(b-a) + sin(c-b) + sin(a-c)|
    """
    n = len(angles)
    idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                idx.append((i,j,k))
    idx = np.array(idx)
    ii, jj, kk = idx[:,0], idx[:,1], idx[:,2]
    
    ai = angles[ii]
    aj = angles[jj]
    ak = angles[kk]
    
    # area = 0.5 * |sin(aj-ai) + sin(ak-aj) + sin(ai-ak)|
    s1 = np.sin(aj - ai)
    s2 = np.sin(ak - aj)
    s3 = np.sin(ai - ak)
    cross_vals = s1 + s2 + s3
    areas = 0.5 * np.abs(cross_vals)
    
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = max_a - tau * np.log(sum_exp)
    w = exp_terms / sum_exp
    
    # Gradient: d(area)/d(angle_i)
    # area = 0.5 * |cross_vals|
    # For convex (positive), d(area)/d(angle_i) = 0.5 * d(cross_vals)/d(angle_i)
    # d(cross_vals)/d(ai) = -cos(aj-ai) + cos(ai-ak) = cos(ai-ak) - cos(aj-ai)
    # d(cross_vals)/d(aj) = cos(aj-ai) - cos(ak-aj)
    # d(cross_vals)/d(ak) = cos(ak-aj) - cos(ai-ak)
    
    c1 = np.cos(aj - ai)
    c2 = np.cos(ak - aj)
    c3 = np.cos(ai - ak)
    
    g_i = c3 - c1
    g_j = c1 - c2
    g_k = c2 - c3
    
    signs = np.sign(cross_vals)
    w_half = 0.5 * w * signs
    
    grad = np.zeros(n)
    np.add.at(grad, ii, w_half * g_i)
    np.add.at(grad, jj, w_half * g_j)
    np.add.at(grad, kk, w_half * g_k)
    
    return lse_val, grad

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

    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau_schedule=None):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth:
                if tau_schedule is not None:
                    _smooth_neg_score_annealed.current_iter = 0
                    def obj_with_annealing(flat):
                        return _smooth_neg_score_annealed(flat, tau_schedule)
                    obj_func = obj_with_annealing
                else:
                    def obj_func(flat):
                        return _smooth_neg_score_and_grad(flat, tau=0.01)[0]
            else:
                obj_func = _neg_score
            
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 25000}
            if method == 'L-BFGS-B':
                opts['maxfev'] = 10000
                opts['maxls'] = 50
            if method == 'Nelder-Mead':
                opts['adaptive'] = True
            
            if use_smooth and tau_schedule is not None:
                def callback(flat):
                    _smooth_neg_score_annealed.current_iter += 1
                    return False
                opts['callback'] = callback
            
            res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    def try_opt_with_grad(x0, method, maxiter, fatol, xatol, tau):
        """L-BFGS-B with analytical gradient."""
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            def obj_and_grad(flat):
                val, grad = _smooth_neg_score_and_grad(flat, tau)
                return val, grad
            
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'maxfev': 10000, 'maxls': 50}
            
            res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                         jac=True, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    def try_circle_opt(angles0, tau=0.01):
        """Optimize angles on unit circle."""
        nonlocal best_score, best_pts
        if left() <= 3.0:
            return
        try:
            def obj_and_grad(ang_flat):
                angles = np.sort(ang_flat)
                lse, grad = _circle_min_area_and_grad(angles, tau)
                return -lse, -grad
            
            res = scipy.optimize.minimize(obj_and_grad, angles0.copy(), method='L-BFGS-B',
                                         jac=True, options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14})
            angles_opt = np.sort(res.x)
            pts = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
            sc = _score(pts)
            if sc > best_score:
                best_score, best_pts = sc, pts.copy()
        except Exception:
            pass

    def try_two_ring_opt(frac0, phase0, tau=0.01):
        """Optimize two-ring config: outer 7 on unit circle, inner 7 at radius frac, phase offset."""
        nonlocal best_score, best_pts
        if left() <= 3.0:
            return
        try:
            n_out = 7
            n_in = 7
            # Parameters: [frac, phase, outer_angles(7), inner_angles(7)]
            def params_to_points(params):
                frac = np.clip(params[0], 0.1, 0.95)
                phase = params[1]
                out_ang = np.sort(params[2:2+n_out])
                in_ang = np.sort(params[2+n_out:2+n_out+n_in]) + phase
                out_pts = np.column_stack([np.cos(out_ang), np.sin(out_ang)])
                in_pts = np.column_stack([frac*np.cos(in_ang), frac*np.sin(in_ang)])
                return np.vstack([out_pts, in_pts])
            
            def obj_and_grad(params):
                pts = params_to_points(params)
                ha = _hull_area(pts)
                if ha < 1e-12:
                    return 1e10, np.zeros(len(params))
                lse, lse_grad = _lse_min_area_and_grad(pts, tau)
                val = -lse / ha
                # Approximate gradient ignoring hull area derivative
                grad_pts = -lse_grad / ha
                # Map back to parameter space (approximate)
                # This is complex; use finite differences for the two-ring
                return val, np.zeros(len(params))
            
            # Use Nelder-Mead for the two-ring since gradient mapping is complex
            def obj(params):
                pts = params_to_points(params)
                return -_score(pts)
            
            x0 = np.concatenate([[frac0, phase0],
                                 np.linspace(0, 2*np.pi, n_out, endpoint=False),
                                 np.linspace(0, 2*np.pi, n_in, endpoint=False)])
            res = scipy.optimize.minimize(obj, x0, method='Nelder-Mead',
                                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            pts_opt = params_to_points(res.x)
            sc = _score(pts_opt)
            if sc > best_score:
                best_score, best_pts = sc, pts_opt.copy()
        except Exception:
            pass

    # Intensive optimization on top candidates
    for pts in top_candidates:
        if left() <= 10.0:
            break
        
        # 1. Nelder-Mead (fast, robust)
        try_opt(pts, 'Nelder-Mead', 20000, 1e-13, 1e-10)
        
        if left() <= 15.0:
            break
            
        # 2. Powell (slower, better local search)
        try_opt(best_pts, 'Powell', 6000, 1e-13, 1e-11)
        
        if left() <= 20.0:
            break

        # 3. L-BFGS-B with analytical gradient (much faster than finite diff)
        try_opt_with_grad(best_pts, 'L-BFGS-B', 5000, 1e-12, 1e-12, tau=0.01)
        
        if left() <= 25.0:
            break

        # 4. Final Polish: Tight Nelder-Mead
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-15, 1e-12)
        
        if left() <= 30.0:
            break

        # 5. Additional polish: Very tight Powell
        try_opt(best_pts, 'Powell', 3000, 1e-14, 1e-13)

    # Circle optimization
    if left() > 15.0:
        for rot in [0.0, np.pi/N, np.pi/7]:
            angles0 = np.linspace(0, 2*np.pi, N, endpoint=False) + rot
            try_circle_opt(angles0, tau=0.01)
            if left() <= 10.0:
                break

    # Two-ring optimization
    if left() > 15.0:
        for frac0 in [0.5, 0.6, 0.7]:
            for phase0 in [0.0, np.pi/7, np.pi/14]:
                try_two_ring_opt(frac0, phase0, tau=0.01)
                if left() <= 10.0:
                    break
            if left() <= 10.0:
                break

    # Final high-precision polish if time allows
    if left() > 15.0:
        # Use smaller tau for more precise min tracking
        try_opt_with_grad(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-14, tau=0.005)
    
    if left() > 10.0:
        try_opt(best_pts, 'Nelder-Mead', 10000, 1e-16, 1e-14)
    
    return _normalize(best_pts)

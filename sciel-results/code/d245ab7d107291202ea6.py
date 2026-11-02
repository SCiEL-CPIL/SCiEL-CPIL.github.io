import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull

N = 14

_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def smooth_score_and_grad(x, beta):
    """
    Returns (-score, -grad_score) using log-sum-exp soft-min with explicit gradient.
    This is the key refinement machinery from the inspiration.
    """
    points = x.reshape((N, 2))
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Handle degenerate cases
    if np.any(areas < 1e-12):
        return 0.0, np.zeros_like(x)
    
    # Soft min via log-sum-exp
    # soft_min = - (1/beta) * log(sum(exp(-beta * areas)))
    # To avoid overflow, use max trick:
    # sum(exp(-beta * areas)) = exp(-beta * max_area) * sum(exp(-beta * (areas - max_area)))
    # soft_min = max_area + (1/beta) * log(sum(exp(-beta * (areas - max_area))))
    # Wait, the standard soft min is:
    # m(x) = - (1/beta) log(sum_i exp(-beta x_i))
    # Let A = areas.
    # m = -1/beta * log(sum exp(-beta A_i))
    # log(sum exp(-beta A_i)) = log(sum exp(-beta (A_i - A_max) + beta A_max))
    # = beta A_max + log(sum exp(-beta (A_i - A_max)))
    # So m = -A_max - (1/beta) log(sum exp(-beta (A_i - A_max)))
    
    A_max = np.max(areas)
    shifted = areas - A_max
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(x)
    
    weights = exp_vals / sum_exp
    
    soft_min_val = -A_max - (1.0 / beta) * np.log(sum_exp)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0, np.zeros_like(x)
    except Exception:
        return 0.0, np.zeros_like(x)
    
    score = soft_min_val / hull_area
    
    # Gradient of soft_min w.r.t. areas
    # d(soft_min)/d(A_i) = -weights_i
    # d(A_i)/d(points)
    # A_i = 0.5 * |cross_i|
    # For differentiability, we assume cross_i > 0 or < 0 and use sign.
    # Actually, since we are maximizing min area, and areas are positive,
    # we can treat |cross| as cross * sign(cross).
    # But near 0, this is problematic. However, beta annealing handles this.
    # Let's use the gradient of cross, multiplied by sign(cross).
    
    signs = np.sign(cross)
    
    # d(cross)/d(p_i)
    # cross = (pjx-pix)(pky-piy) - (pjy-piy)(pkx-pix)
    # d/dpix: -(pky-piy) + (pjy-piy) = v1[1] - v2[1]
    # d/dpiy: (pkx-pix) - (pjx-pix) = v2[0] - v1[0]
    
    # d(cross)/d(pj)
    # d/dpjx: (pky-piy) = v2[1]
    # d/dpjy: -(pkx-pix) = -v2[0]
    
    # d(cross)/d(pk)
    # d/dpkx: -(pjy-piy) = -v1[1]
    # d/dpky: (pjx-pix) = v1[0]
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v2[:, 0] - v1[:, 0]
    
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # d(A_i)/d(p) = 0.5 * sign(cross_i) * d(cross_i)/d(p)
    # d(soft_min)/d(p) = sum_i (-weights_i) * d(A_i)/d(p)
    # d(score)/d(p) = d(soft_min)/d(p) / hull_area  (ignoring hull area gradient for simplicity,
    #          or assuming hull area is locally constant/weakly varying, which is a common approximation
    #          in these optimizations, or we could compute hull gradient but it's complex)
    #          Actually, the inspiration code did: grad_score = grad_points / ha.
    #          This implies it treated ha as constant in the gradient step.
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / hull_area
    
    # We want to minimize -score, so return -score and -grad_score
    return -score, -grad_score.flatten()

def heptagon_7_7_configuration(params):
    r_in, th_off = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    return np.vstack((pts_out, pts_in))

def neg_heptagon_7_7_score(params):
    try:
        points = heptagon_7_7_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def ring_configuration(params):
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    
    while n1 + n2 + n3 > 14:
        if n3 >= 1: n3 -= 1
        elif n2 >= 1: n2 -= 1
        else: n1 -= 1
    while n1 + n2 + n3 < 14:
        n3 += 1
    
    points = []
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    return result

def ring_score(params):
    try:
        points = ring_configuration(params)
        return compute_score(points)
    except Exception:
        return 0.0

def neg_ring_score(params):
    return -ring_score(params)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin for 150s limit
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 0: Fine grid search + local optimization for 7+7 heptagon structure
    if time.time() - start_time < deadline - 50:
        r_vals = np.arange(0.3, 0.86, 0.02)
        th_vals = np.arange(0, np.pi/7 + 1e-9, np.pi/70)
        
        best_77_params = None
        best_77_score = -1e10
        
        for r in r_vals:
            for th in th_vals:
                s = compute_score(heptagon_7_7_configuration([r, th]))
                if s > best_77_score:
                    best_77_score = s
                    best_77_params = [r, th]
        
        if best_77_params is not None:
            x0 = np.array(best_77_params)
            try:
                res = minimize(neg_heptagon_7_7_score, x0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                best_77_points = heptagon_7_7_configuration(res.x)
                try_update(best_77_points)
            except Exception:
                pass
            
            for _ in range(10):
                if time.time() - start_time > deadline - 45:
                    break
                perturbed = best_77_params.copy()
                perturbed[0] += rng.normal(0, 0.03)
                perturbed[1] += rng.normal(0, 0.1)
                perturbed[0] = np.clip(perturbed[0], 0.15, 1.0)
                perturbed[1] = perturbed[1] % (2*np.pi)
                try:
                    res = minimize(neg_heptagon_7_7_score, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                    p = heptagon_7_7_configuration(res.x)
                    try_update(p)
                except Exception:
                    pass

    # Phase 1: Differential Evolution on 3-ring parameters
    bounds = [
        (0.1, 1.5), (0.05, 1.2), (0.0, 0.8),
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),
        (1, 14), (1, 14), (1, 14)
    ]
    
    for restart in range(3):
        if time.time() - start_time > deadline - 35:
            break
        try:
            res = differential_evolution(
                neg_ring_score, bounds,
                seed=rng.integers(0, 2**31),
                maxiter=80, tol=1e-7, popsize=12,
                mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if res.fun < 0:
                points = ring_configuration(res.x)
                try_update(points)
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS-B with soft-min annealing from diverse starts
    if best_points is not None and time.time() - start_time < deadline - 30:
        starts = [best_points.copy()]
        
        for _ in range(8):
            jittered = best_points + rng.normal(0, 0.15, (N, 2))
            starts.append(jittered)
        
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
        a3 = np.linspace(np.pi/10, 2*np.pi + np.pi/10, 5, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.3*np.cos(a3), 0.3*np.sin(a3)))
        )))
        
        a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        a2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 6, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.55*np.cos(a2), 0.55*np.sin(a2)))
        )))
        
        a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        a2 = np.linspace(np.pi/12, 2*np.pi + np.pi/12, 8, endpoint=False)
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.5*np.cos(a2), 0.5*np.sin(a2)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 20:
                break
            
            x0 = start.flatten()
            taus = [0.05, 0.02, 0.005, 0.001, 0.0003, 0.0001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 15:
                    break
                try:
                    res = minimize(
                        soft_min_score, current_x, args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 3: Basin hopping
    if best_points is not None and time.time() - start_time < deadline - 15:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10},
                minsteps=5, T=0.8, stepsize=0.4,
                interval=7, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 4: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 600, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 5: GRADIENT-BASED REFINEMENT (New from Inspiration)
    # This is the critical addition: using the analytic gradient of the soft-min
    # to perform a high-precision local search.
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        
        # Anneal beta from low to high
        betas = [10.0, 50.0, 100.0, 200.0, 400.0, 800.0]
        current_x = x0
        
        for beta in betas:
            if time.time() - start_time > deadline - 2:
                break
            
            try:
                res = minimize(
                    lambda x: smooth_score_and_grad(x, beta),
                    current_x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_x = res.x
                new_p = current_x.reshape((N, 2))
                try_update(new_p)
            except Exception:
                pass

    # Phase 6: Final Nelder-Mead polish
    if best_points is not None and time.time() - start_time < deadline - 2:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 7: Coordinate-wise refinement
    if best_points is not None and time.time() - start_time < deadline - 1:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    return best_points

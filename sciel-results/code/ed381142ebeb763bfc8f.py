import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 140.0  # Wall-clock guard (evaluator kills at 150s)

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

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65]:
        for phase in [0.0, 0.5, 0.25, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
    # Ellipse configurations with varied aspect ratios and rotations
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95)]:
        for rot in [0.0, np.pi / 6, np.pi / 4]:
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + rot
            x = a * np.cos(theta)
            y = b * np.sin(theta)
            pts = np.column_stack([x, y])
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s], [s, c]])
            starts.append(pts @ R.T)
    return starts

def _soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    This is differentiable and approximates -min(A) as tau -> 0.
    """
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
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
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / ha

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
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), g_areas * g_a_x)
    np.add.at(grad, (_I, 1), g_areas * g_a_y)
    np.add.at(grad, (_J, 0), g_areas * g_b_x)
    np.add.at(grad, (_J, 1), g_areas * g_b_y)
    np.add.at(grad, (_K, 0), g_areas * g_c_x)
    np.add.at(grad, (_K, 1), g_areas * g_c_y)
    
    val = -soft_min_val / ha
    grad = -grad / ha
    return val, grad

def _exact_maxmin_obj_grad(points):
    """
    Returns (value, grad) for f(points) = -min_triangle_area(points) / hull_area(points).
    Uses the EXACT minimum (no LSE). The subgradient of min_j a_j is a unit vector on
    the set of minimizers, which is valid for maximizing the minimum.
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

    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    wh = 0.5 * w
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
    
    # 1. Generate and evaluate structured starts
    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        sc = _score(sn)
        scored.append((sc, sn))
        if sc > best_score:
            best_score = sc
            best_pts = sn.copy()
    
    scored.sort(key=lambda x: -x[0])
    
    # 2. Global Search: Differential Evolution
    if left() > 40.0:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            # Use a mix of top structured starts and random points for initial population
            pop_size = 30
            x0 = []
            for i in range(pop_size):
                if i < len(scored):
                    x0.append(scored[i][1].flatten())
                else:
                    x0.append(np.random.uniform(-1.0, 1.0, 2 * N))
            
            # DE requires an objective function that returns a scalar
            def de_obj(x):
                return _score(x.reshape(-1, 2))
            
            # Note: scipy DE maximizes by default if we pass a negative or use 'polish'
            # Standard DE in scipy minimizes. So we minimize -score.
            def de_neg_obj(x):
                return -_score(x.reshape(-1, 2))
            
            res = scipy.optimize.differential_evolution(
                de_neg_obj, bounds, maxiter=200, tol=1e-8, 
                popsize=15, seed=42, polish=True, init='sobol'
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            try_update(opt_pts)
        except Exception:
            pass
    
    # 3. Local Refinement: L-BFGS-B with Soft Min (Differentiable)
    # We refine the current best point using the smooth surrogate
    if left() > 10.0:
        try:
            # Start from best_pts
            x0 = best_pts.flatten()
            
            # Annueling tau: start with larger tau (smoother, better landscape) and decrease
            taus = [0.1, 0.05, 0.02, 0.01, 0.005]
            current_pts = best_pts.copy()
            
            for tau in taus:
                if left() <= 5.0:
                    break
                try:
                    # Define objective and gradient for current tau
                    def obj_grad(x):
                        v, g = _soft_min_score_grad(x, tau)
                        return v, g
                    
                    res = scipy.optimize.minimize(
                        obj_grad, current_pts.flatten(), method='L-BFGS-B',
                        jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    opt_pts = _normalize(res.x.reshape(-1, 2))
                    try_update(opt_pts)
                    current_pts = best_pts.copy() # Use the improved best for next tau
                except Exception:
                    pass
        except Exception:
            pass

    # 4. Local Refinement: Exact Subgradient / Nelder-Mead
    # If we have time, do a final polish with non-smooth methods on the best found
    if left() > 10.0:
        try:
            # Nelder-Mead is robust to non-smoothness
            def nm_obj(x):
                return -_score(x.reshape(-1, 2))
            
            res = scipy.optimize.minimize(
                nm_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            try_update(opt_pts)
        except Exception:
            pass

    # 5. Final Check
    return _normalize(best_pts)

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")

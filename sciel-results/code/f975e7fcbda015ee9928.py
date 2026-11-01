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

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65]:
        for phase in [0.0, 0.5, 0.25, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
    return starts

def _lse_min_area(points, tau=0.01):
    """
    Smooth approximation of min triangle area using Log-Sum-Exp.
    Lower tau -> closer to true min.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # LSE: min(x) ~ -tau * log(sum(exp(-x/tau)))
    # We want to maximize min area, so minimize -min area.
    # -min(areas) ~ tau * log(sum(exp(-areas/tau)))
    max_a = np.max(areas)
    # Shift for numerical stability
    exp_terms = np.exp(-(areas - max_a) / tau)
    lse_val = max_a + tau * np.log(np.sum(exp_terms))
    return lse_val

def _smooth_neg_score(flat, tau=0.01):
    """
    Smooth objective for L-BFGS-B.
    Returns negative of (LSE_min_area / Hull_Area).
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    min_a_lse = _lse_min_area(pts, tau)
    # We want to maximize min_a / ha
    # So minimize -min_a / ha
    return -min_a_lse / ha

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

    # Top candidates for intensive optimization
    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            obj_func = _smooth_neg_score if use_smooth else _neg_score
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 25000}
            if method == 'L-BFGS-B':
                opts['maxfev'] = 5000 # Limit finite diff calls
                opts['maxls'] = 50
            res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
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

        # 3. L-BFGS-B with smooth surrogate (potential for faster convergence if gradient is good)
        try_opt(best_pts, 'L-BFGS-B', 1000, 1e-12, 1e-12, use_smooth=True)
        
        if left() <= 25.0:
            break

        # 4. Final Polish: Tight Nelder-Mead
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-15, 1e-12)

    return _normalize(best_pts)

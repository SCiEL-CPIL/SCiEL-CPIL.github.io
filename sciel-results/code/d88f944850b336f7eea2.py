import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 140.0  # wall-clock guard

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
    # Regular polygon baseline
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3, np.pi / 7]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    # Two-ring configurations
    for frac in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
            
    # Elliptical hulls
    for a in [0.9, 1.0, 1.1]:
        for b in [0.7, 0.8, 0.9, 1.0]:
            for rot in [0.0, np.pi/6, np.pi/4, np.pi/3]:
                starts.append(_ellipse_config(N, a, b, rot))
                
    # Three-ring configuration
    n1, n2, n3 = 5, 5, 4
    for r1 in [0.3, 0.4]:
        for r2 in [0.6, 0.7]:
            ring1 = _regular_polygon(n1, 1.0, 0.0)
            ring2 = _regular_polygon(n2, r2, np.pi/n2)
            ring3 = _regular_polygon(n3, r1, np.pi/n3)
            starts.append(np.vstack([ring1, ring2, ring3]))
            
    return starts

def _lse_min_area(points, tau=0.01):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    exp_terms = np.exp(-(areas - min_a) / tau)
    lse_val = -tau * np.log(np.sum(exp_terms)) + min_a
    return lse_val

def _smooth_neg_score(flat, tau=0.01):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    min_a_soft = _lse_min_area(pts, tau)
    return -min_a_soft / ha

def construct():
    t0 = time.time()
    def left():
        return DEADLINE - (time.time() - t0)

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

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau=0.01):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth:
                obj_func = lambda f: _smooth_neg_score(f, tau)
            else:
                obj_func = _neg_score
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 25000}
            if method == 'L-BFGS-B':
                opts['maxfev'] = 5000
                opts['maxls'] = 50
            res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Phase 1: Nelder-Mead on top candidates
    for pts in top_candidates:
        if left() <= 12.0:
            break
        try_opt(pts, 'Nelder-Mead', 15000, 1e-13, 1e-10)

    # Phase 2: L-BFGS-B with smooth surrogate on best (coarse tau first)
    if left() > 20.0:
        for tau in [0.1, 0.05]:
            if left() <= 8.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 800, 1e-12, 1e-12, use_smooth=True, tau=tau)

    # Phase 3: Powell on best
    if left() > 18.0:
        try_opt(best_pts, 'Powell', 5000, 1e-13, 1e-11)

    # Phase 4: L-BFGS-B with tight tau for precision
    if left() > 12.0:
        for tau in [0.02, 0.01, 0.005]:
            if left() <= 5.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 600, 1e-14, 1e-14, use_smooth=True, tau=tau)

    # Phase 5: Final Polish with tight Nelder-Mead
    if left() > 5.0:
        try_opt(best_pts, 'Nelder-Mead', 8000, 1e-15, 1e-13)

    return _normalize(best_pts)

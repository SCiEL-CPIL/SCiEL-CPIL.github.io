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

def _hull_area_fast(points):
    """Convex-hull area via shoelace on the hull boundary (fast, no re-hull)."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        verts = hull.vertices
        p = points[verts]
        x = p[:, 0]
        y = p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    except Exception:
        return _hull_area(points)

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
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 7, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    for frac in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        for phase in [0.0, 0.5, 0.25, 0.75, 1 / 7, 2 / 7]:
            starts.append(_two_ring_config(N, frac, phase))
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95)]:
        for rot in [0.0, np.pi / 6, np.pi / 4]:
            starts.append(_ellipse_config(N, a, b, rot))
    return starts

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

    ha = _hull_area_fast(points)
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

def _circle_min_area_grad(angles):
    """
    For 14 points on the unit circle at the given (sorted) angles, return
    (value, grad) for f = -min_triangle_area. Triangle area for points at angles
    a,b,c on the unit circle is 0.5*|sin(b-a)+sin(c-b)+sin(a-c)|.
    """
    n = len(angles)
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    idx = np.array(idx)
    ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]

    ai = angles[ii]
    aj = angles[jj]
    ak = angles[kk]
    cross_vals = np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak)
    areas = 0.5 * np.abs(cross_vals)

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(len(idx))
    w[mask] = 1.0 / cnt

    c1 = np.cos(aj - ai)
    c2 = np.cos(ak - aj)
    c3 = np.cos(ai - ak)
    g_i = c3 - c1
    g_j = c1 - c2
    g_k = c2 - c3

    signs = np.sign(cross_vals)
    wh = 0.5 * w * signs
    grad = np.zeros(n)
    np.add.at(grad, ii, wh * g_i)
    np.add.at(grad, jj, wh * g_j)
    np.add.at(grad, kk, wh * g_k)

    return -m, -grad

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

    top_candidates = [pts for _, pts in scored[:8]]

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
                        'maxfev': 15000, 'maxls': 40}
                res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                              jac=True, options=opts)
            else:
                opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 20000}
                if method == 'Nelder-Mead':
                    opts['adaptive'] = True
                res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    def try_circle_opt(angles0):
        nonlocal best_score, best_pts
        if left() <= 3.0:
            return
        try:
            def obj_and_grad(ang_flat):
                angles = np.sort(ang_flat)
                return _circle_min_area_grad(angles)
            res = scipy.optimize.minimize(obj_and_grad, angles0.copy(), method='L-BFGS-B',
                                          jac=True, options={'maxiter': 2000,
                                                             'fatol': 1e-14, 'xatol': 1e-14})
            angles_opt = np.sort(res.x)
            pts = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
            sc = _score(pts)
            if sc > best_score:
                best_score, best_pts = sc, pts.copy()
        except Exception:
            pass

    # ---- Phase 1: coarse search over diverse starts ----
    for pts in top_candidates:
        if left() <= 12.0:
            break
        try_opt(pts, 'Nelder-Mead', 8000, 1e-12, 1e-9)
        if left() <= 16.0:
            break
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-12, 1e-11, use_grad=True)

    # ---- Angular packing on the circle (all points on boundary) ----
    if left() > 15.0:
        for rot in [0.0, np.pi / N, np.pi / 7, np.pi / 14, np.pi / 10]:
            angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False) + rot
            try_circle_opt(angles0)
            if left() <= 10.0:
                break

    # ---- Phase 2: fine polish on the best configuration ----
    if left() > 12.0:
        try_opt(best_pts, 'L-BFGS-B', 4000, 1e-13, 1e-12, use_grad=True)
    if left() > 10.0:
        try_opt(best_pts, 'Nelder-Mead', 6000, 1e-14, 1e-11)
    if left() > 8.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-13, use_grad=True)
    if left() > 6.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-15, 1e-13)

    return _normalize(best_pts)

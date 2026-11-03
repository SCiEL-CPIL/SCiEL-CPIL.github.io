"""Hexagon packing: 11 unit hexagons in a minimal regular hexagon.
Strong rewrite: exact O(1) enclosing-hex center (no LP), correct
separation test (vertex + edge-edge), and Nelder-Mead + exact polish."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0   # safety margin inside 150 s wall-clock
SQ3 = math.sqrt(3.0)
A60 = math.radians(60.0)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_verts_array(x, y, ang):
    """Vertices of unit hexagon (center (x,y), rotation ang radians). (6,2)."""
    t = ang + np.arange(6) * A60
    return np.column_stack([x + np.cos(t), y + np.sin(t)])

def _all_verts(inner):
    """Stack all inner hexagon vertices -> (6*N, 2)."""
    return np.vstack([_hex_verts_array(x, y, math.radians(a)) for x, y, a in inner])

def _face_normals(theta):
    """Outward unit normals of the 6 faces of a hexagon rotated by theta (rad)."""
    t = theta + math.radians(30.0) + np.arange(6) * A60
    return np.column_stack([np.cos(t), np.sin(t)])

def _min_enclosing_exact(verts, theta):
    """Exact min enclosing regular hexagon (rot theta) for point set verts.
    Returns (cx, cy, side). O(N) -- no LP."""
    n = _face_normals(theta)
    P = verts @ n.T                       # (6N, 6) projections
    hi = P.max(axis=0)
    lo = P.min(axis=0)
    c = 0.5 * (hi + lo)                   # center projection on each normal
    t = float(np.max(hi - lo) / 2.0)      # required inradius
    side = t / (SQ3 / 2.0)
    # de-project center
    M = n.T @ n                           # 6x6
    try:
        cc = np.linalg.solve(M, c)
    except np.linalg.LinAlgError:
        cc = np.linalg.lstsq(M, c, rcond=None)[0]
    return float(cc[0]), float(cc[1]), side

def _side_at(verts, theta):
    return _min_enclosing_exact(verts, theta)[2]

def _min_enclosing_angle(verts, angles):
    """Best (cx,cy,side,theta) over a list of angles."""
    best = (0.0, 0.0, float('inf'), 0.0)
    for th in angles:
        th = float(th)
        cx, cy, side = _min_enclosing_exact(verts, th)
        if side < best[2]:
            best = (cx, cy, side, th)
    return best

# ---------------------------------------------------------------------------
# Separation (correct: vertex-in + edge-edge)
# ---------------------------------------------------------------------------
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def _on_seg(a, b, c, eps):
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

def _seg_cross(p1, p2, p3, p4, eps=1e-10):
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_seg(p3, p4, p1, eps): return True
    if abs(d2) <= eps and _on_seg(p3, p4, p2, eps): return True
    if abs(d3) <= eps and _on_seg(p1, p2, p3, eps): return True
    if abs(d4) <= eps and _on_seg(p1, p2, p4, eps): return True
    return False

def _point_in_hex(p, poly, eps=1e-10):
    for i in range(6):
        a = poly[i]; b = poly[(i + 1) % 6]
        if _orient(a, b, p) > eps:
            return False
    return True

def _hex_overlap(poly1, poly2, eps=1e-10):
    for v in poly1:
        if _point_in_hex(v, poly2, eps):
            return True
    for v in poly2:
        if _point_in_hex(v, poly1, eps):
            return True
    for i in range(6):
        for j in range(6):
            if _seg_cross(poly1[i], poly1[(i + 1) % 6],
                          poly2[j], poly2[(j + 1) % 6], eps):
                return True
    return False

def _check_disjoint(inner, eps=1e-10):
    polys = [_hex_verts_array(x, y, math.radians(a)).tolist() for x, y, a in inner]
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(polys[i], polys[j], eps):
                return False
    return True

# ---------------------------------------------------------------------------
# Smooth surrogate (log-sum-exp) for gradient-free + gradient optimizers
# ---------------------------------------------------------------------------
def _smooth_sep(inner, beta=60.0):
    req = SQ3 - 0.002
    v = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            v.append(req - math.hypot(dx, dy))
    pos = [x for x in v if x > 0]
    if not pos:
        return 0.0
    m = max(beta * x for x in pos)
    s = sum(math.exp(beta * x - m) for x in pos)
    return (m + math.log(s)) / beta

def _soft_side(inner, theta, beta=40.0):
    verts = _all_verts(inner)
    n = _face_normals(theta)
    P = verts @ n.T
    hi = P.max(axis=0)
    lo = P.min(axis=0)
    terms = [beta * (hi[k] - lo[k]) / 2.0 for k in range(6)]
    m = max(terms)
    s = sum(math.exp(t - m) for t in terms)
    return (m + math.log(s)) / beta / (SQ3 / 2.0)

def _objective(inner, theta, sep_beta=60.0, side_beta=40.0):
    return _soft_side(inner, theta, side_beta) + 6.0 * _smooth_sep(inner, sep_beta)

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def _nm_optimize(inner, theta, max_iter=2500, tol=1e-12):
    """Nelder-Mead on positions+angles, smooth objective."""
    from scipy.optimize import minimize
    def obj(p):
        inn = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
        return _objective(inn, theta, 60.0, 40.0)
    x0 = np.array([c for h in inner for c in h], dtype=float)
    try:
        r = minimize(obj, x0, method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol,
                              'adaptive': True, 'disp': False})
        if r.fun < 100.0:
            out = [[float(r.x[3*i]), float(r.x[3*i+1]), float(r.x[3*i+2])] for i in range(N)]
            if _check_disjoint(out):
                return out
    except Exception:
        pass
    return inner

def _slsqp_optimize(inner, theta, max_iter=120):
    """SLSQP with explicit smooth separation constraints."""
    from scipy.optimize import minimize
    def obj(p):
        inn = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
        return _soft_side(inn, theta, 40.0)
    cons = []
    req = SQ3 - 0.002
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda p, i=i, j=j:
                             math.hypot(p[3*i]-p[3*j], p[3*i+1]-p[3*j+1]) - req})
    x0 = np.array([c for h in inner for c in h], dtype=float)
    try:
        r = minimize(obj, x0, method='SLSQP', bounds=[(None, None)] * (3 * N),
                     constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-10,
                                                 'disp': False})
        if r.fun < 100.0:
            out = [[float(r.x[3*i]), float(r.x[3*i+1]), float(r.x[3*i+2])] for i in range(N)]
            if _check_disjoint(out):
                return out
    except Exception:
        pass
    return inner

def _exact_polish(inner, theta, max_iter=150):
    """Exact piecewise-linear coordinate descent (no LP)."""
    best = [h[:] for h in inner]
    verts = _all_verts(best)
    _, _, best_side = _min_enclosing_exact(verts, theta)
    step = 0.02
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 8.0:
            break
        improved = False
        for i in range(N):
            for k in range(3):
                for d in (-1, 1):
                    tr = [h[:] for h in best]
                    tr[i][k] += d * step
                    if _check_disjoint(tr):
                        vs = _all_verts(tr)
                        _, _, side = _min_enclosing_exact(vs, theta)
                        if side < best_side - 1e-12:
                            best = tr; verts = vs; best_side = side; improved = True
        if not improved:
            step *= 0.6
            if step < 1e-7:
                break
    return best

def _refine_angle(verts, angle, coarse=0.5, fine=0.05):
    best = _min_enclosing_exact(verts, angle)
    for da in np.arange(-coarse * 6, coarse * 6 + 1e-9, coarse):
        th = (angle + float(da)) % (2 * math.pi)
        r = _min_enclosing_exact(verts, th)
        if r[2] < best[2]:
            best = (r[0], r[1], r[2], th)
    for da in np.arange(-coarse, coarse + 1e-9, fine):
        th = (best[3] + float(da)) % (2 * math.pi)
        r = _min_enclosing_exact(verts, th)
        if r[2] < best[2]:
            best = (r[0], r[1], r[2], th)
    return best

# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------
def _tri_lattice_positions(scale=1.002, ang=30.0):
    u = (SQ3 * scale, 0.0)
    v = (SQ3 / 2 * scale, 1.5 * scale)
    cand = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            cand.append((x * x + y * y, x, y))
    cand.sort(key=lambda t: t[0])
    return [[x, y, ang] for _, x, y in cand[:N]]

def _ring_positions(scale=1.002, ang=30.0):
    pos = [[0.0, 0.0, ang]]
    for k in range(6):
        a = A60 * k
        pos.append([SQ3 * scale * math.cos(a), SQ3 * scale * math.sin(a), ang])
    r2 = 2 * SQ3 * scale
    for k in (0, 2, 4, 6):
        a = A60 * k + math.radians(30.0)
        pos.append([r2 * math.cos(a), r2 * math.sin(a), ang])
    return pos[:N]

def _compact_positions(scale=1.002, ang=30.0):
    pos = [[0.0, 0.0, ang]]
    for k in range(6):
        a = A60 * k
        pos.append([SQ3 * scale * math.cos(a), SQ3 * scale * math.sin(a), ang])
    r2 = 3.0 * scale
    for k in (0, 2, 4, 6):
        a = A60 * k
        pos.append([r2 * math.cos(a), r2 * math.sin(a), ang])
    return pos[:N]

# ---------------------------------------------------------------------------
def construct():
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    generators = [_tri_lattice_positions, _ring_positions, _compact_positions]
    scales = [1.002, 1.006]
    angles = [30.0, 0.0]
    configs = []
    for gen in generators:
        for s in scales:
            for a in angles:
                c = gen(scale=s, ang=a)
                if _check_disjoint(c):
                    configs.append(c)
    if not configs:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    coarse = [math.radians(d) for d in range(0, 60, 5)]
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # 1) evaluate every config with exact enclosing + angle search
    for c in configs:
        if time.time() > _DEADLINE - 20.0:
            break
        verts = _all_verts(c)
        cx, cy, side, ang = _min_enclosing_angle(verts, coarse)
        if side < best_side:
            best_side, best_inner, best_center, best_angle = side, [h[:] for h in c], (cx, cy), ang

    if best_inner is None:
        best_inner = configs[0]
        verts = _all_verts(best_inner)
        cx, cy, best_side, best_angle = _min_enclosing_angle(verts, coarse)
        best_center = (cx, cy)

    # 2) Nelder-Mead on the best config
    if time.time() < _DEADLINE - 40.0:
        polished = _nm_optimize(best_inner, best_angle)
        if polished is not best_inner:
            verts = _all_verts(polished)
            cx, cy, side, ang = _refine_angle(verts, best_angle)
            if side < best_side - 1e-9:
                best_side, best_inner, best_center, best_angle = side, polished, (cx, cy), ang

    # 3) SLSQP on best (and a second) config
    if time.time() < _DEADLINE - 35.0:
        polished = _slsqp_optimize(best_inner, best_angle)
        if polished is not best_inner:
            verts = _all_verts(polished)
            cx, cy, side, ang = _refine_angle(verts, best_angle)
            if side < best_side - 1e-9:
                best_side, best_inner, best_center, best_angle = side, polished, (cx, cy), ang
    if len(configs) > 1 and time.time() < _DEADLINE - 30.0:
        c2 = next((c for c in configs if c is not best_inner), None)
        if c2 is not None:
            polished = _slsqp_optimize(c2, best_angle)
            if _check_disjoint(polished):
                verts = _all_verts(polished)
                cx, cy, side, ang = _refine_angle(verts, best_angle)
                if side < best_side - 1e-9:
                    best_side, best_inner, best_center, best_angle = side, polished, (cx, cy), ang

    # 4) exact piecewise-linear polish (no LP)
    if time.time() < _DEADLINE - 20.0:
        polished = _exact_polish(best_inner, best_angle, max_iter=120)
        if polished is not best_inner:
            verts = _all_verts(polished)
            cx, cy, side, ang = _refine_angle(verts, best_angle)
            if side < best_side - 1e-12:
                best_side, best_inner, best_center, best_angle = side, polished, (cx, cy), ang

    # 5) final exact evaluation with fine angle refinement
    if time.time() < _DEADLINE - 5.0:
        verts = _all_verts(best_inner)
        cx, cy, side, ang = _refine_angle(verts, best_angle, coarse=1.0, fine=0.02)
        best_side, best_center, best_angle = side, (cx, cy), ang

    return best_inner, (best_center[0], best_center[1]), best_side, math.degrees(best_angle) % 360.0
# EVOLVE-BLOCK-END

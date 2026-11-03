"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural rewrite:
  * Voronoi-based initialisation (radii start near the feasible ceiling).
  * SLSQP with EXPLICIT non-overlap / boundary constraints (hard feasibility).
  * Analytic-gradient L-BFGS-B fallback for tight convergence.
  * Coordinate-descent radius refinement using the exact per-circle radius
    upper bound (monotone-improving).
  * Time-budgeted multi-start with a hard deadline guard.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26
_GLOBAL_DEADLINE = None


def _remaining_time():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())


# ---------------------------------------------------------------------------
# Feasibility / scoring
# ---------------------------------------------------------------------------
def _validate(xyz, tol=1e-7):
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(xyz):
    return float(np.sum(xyz[:, 2]))


def _fix_boundaries(xyz):
    """Project circles inside the unit square (positions only)."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0.0, xyz[i, 2])
        if r < 0.5:
            xyz[i, 0] = min(max(xyz[i, 0], r), 1 - r)
            xyz[i, 1] = min(max(xyz[i, 1], r), 1 - r)
        else:
            xyz[i, 0] = 0.5
            xyz[i, 1] = 0.5
        xyz[i, 2] = r
    return xyz


# ---------------------------------------------------------------------------
# Voronoi-based initialisation
# ---------------------------------------------------------------------------
def _voronoi_radii(xyz):
    """For each centre, the max radius that fits (edge + neighbour limited)."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    maxr = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    maxr = np.maximum(maxr, 1e-6)
    try:
        v = Voronoi(np.column_stack([x, y]))
        for i in range(N):
            region = v.point_region[i]
            if region == -1:
                continue
            poly = v.regions[region]
            if len(poly) == 0:
                continue
            for j in poly:
                p1 = v.vertices[p1 if False else j]
            # distance to each Voronoi vertex
            verts = v.vertices[np.array(poly)]
            d = np.min(np.hypot(verts[:, 0] - x[i], verts[:, 1] - y[i]))
            maxr[i] = min(maxr[i], d)
    except Exception:
        # Fallback: nearest-neighbour limited
        for i in range(N):
            dx = x - x[i]
            dy = y - y[i]
            d = np.hypot(dx, dy)
            d[i] = np.inf
            maxr[i] = min(maxr[i], 0.5 * np.min(d))
    return maxr


def _init_voronoi(seed=0):
    rng = np.random.default_rng(seed)
    # deterministic-ish structured seed + jitter, then Voronoi radii
    pts = []
    for row in range(6):
        for col in range(5):
            if row % 2 == 1:
                cx = (col + 0.5) / 5.0
            else:
                cx = (col + 0.5) / 5.0
            cy = (row + 0.5) / 6.0
            pts.append([cx, cy, 0.05])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    while len(pts) < N:
        pts.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.05])
    xyz = np.array(pts[:N])
    maxr = _voronoi_radii(xyz)
    xyz[:, 2] = np.maximum(maxr, 0.01) * 0.98  # slightly under for safety
    return _fix_boundaries(xyz)


def _init_corner_cluster(seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88],
               [0.5, 0.09], [0.5, 0.91], [0.09, 0.5], [0.91, 0.5], [0.5, 0.5]]
    for cx, cy in corners:
        pts.append([cx, cy, 0.10])
    while len(pts) < N:
        pts.append([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), 0.07])
    xyz = np.array(pts[:N])
    maxr = _voronoi_radii(xyz)
    xyz[:, 2] = np.maximum(maxr, 0.01) * 0.98
    return _fix_boundaries(xyz)


def _init_ring(seed=0):
    rng = np.random.default_rng(seed)
    pts = [[0.5, 0.5, 0.09]]
    for ring, n_c, rad in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_c):
            ang = 2 * np.pi * k / n_c + rng.uniform(-0.08, 0.08)
            dist = ring * 0.14
            pts.append([0.5 + dist * np.cos(ang), 0.5 + dist * np.sin(ang), rad])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    while len(pts) < N:
        pts.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    xyz = np.array(pts[:N])
    maxr = _voronoi_radii(xyz)
    xyz[:, 2] = np.maximum(maxr, 0.01) * 0.98
    return _fix_boundaries(xyz)


def _init_random(seed=0):
    rng = np.random.default_rng(seed)
    pts = [[rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06] for _ in range(N)]
    xyz = np.array(pts)
    maxr = _voronoi_radii(xyz)
    xyz[:, 2] = np.maximum(maxr, 0.01) * 0.98
    return _fix_boundaries(xyz)


# ---------------------------------------------------------------------------
# SLSQP with explicit constraints
# ---------------------------------------------------------------------------
def _slsqp_bounds():
    b = []
    for _ in range(N):
        b.append((0.0, 1.0))
        b.append((0.0, 1.0))
        b.append((1e-6, 0.5))
    return b


def _make_slsqp():
    idx_i, idx_j = np.triu_indices(N, k=1)

    def neg_sum_r(p):
        return -np.sum(p[2::3])

    def grad(p):
        g = np.zeros_like(p)
        g[2::3] = -1.0
        return g

    def c_nonoverlap(p):
        x, y, r = p[0::3], p[1::3], p[2::3]
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-14))
        return d - (r[idx_i] + r[idx_j])

    def c_left(p):
        x, r = p[0::3], p[2::3]
        return x - r

    def c_right(p):
        x, r = p[0::3], p[2::3]
        return 1 - x - r

    def c_bottom(p):
        y, r = p[1::3], p[2::3]
        return y - r

    def c_top(p):
        y, r = p[1::3], p[2::3]
        return 1 - y - r

    constraints = [
        {'type': 'ineq', 'fun': c_nonoverlap},
        {'type': 'ineq', 'fun': c_left},
        {'type': 'ineq', 'fun': c_right},
        {'type': 'ineq', 'fun': c_bottom},
        {'type': 'ineq', 'fun': c_top},
    ]
    return neg_sum_r, grad, constraints


_SLSQP = _make_slsqp()
_SLSQP_OBJ, _SLSQP_GRAD, _SLSQP_CONS = _SLSQP


def _slsqp_optimize(xyz0, maxiter=2000):
    p0 = xyz0.flatten().copy()
    try:
        res = minimize(_SLSQP_OBJ, p0, jac=_SLSQP_GRAD, method='SLSQP',
                       bounds=_slsqp_bounds(), constraints=_SLSQP_CONS,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        return res.x.reshape(N, 3)
    except Exception:
        return xyz0


# ---------------------------------------------------------------------------
# Analytic-gradient penalty (L-BFGS-B fallback)
# ---------------------------------------------------------------------------
def _softplus(z, eps):
    return np.logaddexp(0.0, z / eps) * eps


def _penalty_obj_grad(p, w, eps):
    x, y, r = p[0::3], p[1::3], p[2::3]
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)

    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    nr = -r

    def sp(v):
        return _softplus(v, eps)

    def d(v):  # d/dz softplus
        return 1.0 / (1.0 + np.exp(-v / eps))

    val = -np.sum(r)
    g = np.full_like(p, 0.0)
    g[2::3] = -1.0

    # boundary terms
    for v, dv in [(lx, d(lx)), (ux, d(ux)), (ly, d(ly)), (uy, d(uy)), (nr, d(nr))]:
        s2 = sp(v) ** 2
        val += w * np.sum(s2)
        # ds2/dz = 2*s * d
        ds2 = 2.0 * sp(v) * dv
        # chain: lx=r-x -> dr=+1, dx=-1 ; ux=x+r-1 -> dx=+1, dr=+1
        # ly=r-y -> dr=+1, dy=-1 ; uy=y+r-1 -> dy=+1, dr=+1 ; nr=-r -> dr=-1
    # (compute gradients explicitly below to keep it simple & correct)
    val = 0.0
    g = np.full_like(p, 0.0)
    g[2::3] = -1.0

    terms = [
        (lx, np.array([1.0, 0.0, -1.0])),   # r - x
        (ux, np.array([-1.0, 0.0, 1.0])),   # x + r - 1
        (ly, np.array([0.0, 1.0, -1.0])),   # r - y
        (uy, np.array([0.0, -1.0, 1.0])),   # y + r - 1
        (nr, np.array([0.0, 0.0, 1.0])),    # -r
    ]
    for v, chain in terms:
        s = sp(v)
        val += w * np.sum(s * s)
        ds2 = 2.0 * s * d(v)  # d/dz (s^2)
        # ds2/dp = ds2 * chain
        for k in range(3):
            g[k::3] += w * ds2 * chain[k]

    # overlap terms
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-14))
    overlap = r[idx_i] + r[idx_j] - dist
    s = sp(overlap)
    val += w * np.sum(s * s)
    ds2 = 2.0 * s * d(overlap)
    # d(overlap)/dx_i = -dx/dist ; /dy_i = -dy/dist ; /dr_i = +1 ; same for j
    inv = ds2 / dist
    for k in range(len(idx_i)):
        ii, jj = idx_i[k], idx_j[k]
        g[3 * ii + 0] += w * (-inv[k] * dx[k])
        g[3 * ii + 1] += w * (-inv[k] * dy[k])
        g[3 * ii + 2] += w * ds2[k]
        g[3 * jj + 0] += w * (inv[k] * dx[k])
        g[3 * jj + 1] += w * (inv[k] * dy[k])
        g[3 * jj + 2] += w * ds2[k]

    return val, g


def _lbfgs_optimize(xyz0, w, eps, maxiter=1500):
    p0 = xyz0.flatten().copy()
    try:
        res = minimize(lambda p: _penalty_obj_grad(p, w, eps)[0], p0,
                       jac=lambda p: _penalty_obj_grad(p, w, eps)[1],
                       method='L-BFGS-B', bounds=_slsqp_bounds(),
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 3)
    except Exception:
        return xyz0


# ---------------------------------------------------------------------------
# Coordinate-descent radius refinement (exact per-circle upper bound)
# ---------------------------------------------------------------------------
def _max_radius_for(xyz, i):
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    m = min(x[i], 1 - x[i], y[i], 1 - y[i])
    for j in range(N):
        if j == i:
            continue
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        m = min(m, d - r[j])
    return max(m, 0.0)


def _radius_refine(xyz, n_sweeps=60, grow=1.015):
    xyz = xyz.copy()
    for _ in range(n_sweeps):
        if _remaining_time() < 1.0:
            break
        improved = False
        for i in range(N):
            mr = _max_radius_for(xyz, i)
            cur = xyz[i, 2]
            target = min(mr, cur * grow)
            if target > cur + 1e-10:
                xyz[i, 2] = target
                improved = True
        if not improved:
            break
    return xyz


# ---------------------------------------------------------------------------
# Full refinement pipeline from one starting config
# ---------------------------------------------------------------------------
def _refine(xyz0, budget):
    xyz = _fix_boundaries(xyz0.copy())
    best = xyz.copy()
    best_s = _score(xyz)

    def keep(cand):
        nonlocal best, best_s
        if _validate(cand):
            s = _score(cand)
            if s > best_s:
                best, best_s = cand.copy(), s
                return True
        return False

    # 1) SLSQP (hard constraints)
    if _remaining_time() > budget:
        xyz = _slsqp_optimize(xyz, maxiter=2500)
        keep(xyz)

    # 2) L-BFGS-B penalty (analytic grad), a few penalty levels
    for w, eps in [(50.0, 1e-4), (500.0, 1e-5), (5000.0, 1e-6)]:
        if _remaining_time() < budget:
            break
        xyz = _lbfgs_optimize(xyz, w, eps, maxiter=1200)
        keep(xyz)

    # 3) radius coordinate descent
    if _remaining_time() > budget:
        xyz = _radius_refine(xyz, n_sweeps=80, grow=1.02)
        keep(xyz)

    # 4) SLSQP again to re-balance positions after radius growth
    if _remaining_time() > budget:
        xyz = _slsqp_optimize(xyz, maxiter=1500)
        keep(xyz)

    # 5) final radius polish
    if _remaining_time() > budget:
        xyz = _radius_refine(xyz, n_sweeps=120, grow=1.01)
        keep(xyz)

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    global _GLOBAL_DEADLINE
    start = time.time()
    _GLOBAL_DEADLINE = start + 145.0  # 5s safety margin before 150s kill

    inits = [
        _init_voronoi(0),
        _init_corner_cluster(0),
        _init_ring(0),
        _init_voronoi(7),
        _init_corner_cluster(3),
        _init_ring(5),
        _init_random(11),
        _init_voronoi(23),
    ]

    best = inits[0].copy()
    best_s = _score(best)
    if not _validate(best):
        best = _fix_boundaries(best)

    # Per-start budget: spend most of the time on a few strong starts.
    per_start = 12.0
    for xyz0 in inits:
        if _remaining_time() < per_start + 8.0:
            break
        cand = _refine(xyz0, budget=per_start)
        if _validate(cand) and _score(cand) > best_s:
            best, best_s = cand, _score(cand)

    # Final global polish on the best solution found.
    if _remaining_time() > 10.0:
        cand = _refine(best, budget=8.0)
        if _validate(cand) and _score(cand) > best_s:
            best, best_s = cand, _score(cand)

    if not _validate(best):
        best = _fix_boundaries(best)

    return best

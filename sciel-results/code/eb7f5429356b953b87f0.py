# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(30.0 * np.arange(12))
    return np.stack([np.cos(base), np.sin(base)], axis=1)


def _pair_gap(c1, a1, c2, a2):
    n1 = _hex_normals(a1)
    d = c2 - c1
    proj = n1 @ d - 2.0
    return float(np.max(proj))


def _min_pair_gap(c, a):
    best = np.inf
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            g = _pair_gap(c[i], a[i], c[j], a[j])
            if g < best:
                best = g
    return best


def _contain_violation(p, R):
    """p: (2,) relative to outer center. R: outer circumradius.
    Returns max(0, amount by which p is outside a flat-top hexagon of circumradius R)."""
    x, y = p
    c1 = abs(x) - R
    c2 = abs(0.5 * x + np.sqrt(3) / 2 * y) - np.sqrt(3) / 2 * R
    c3 = abs(-0.5 * x + np.sqrt(3) / 2 * y) - np.sqrt(3) / 2 * R
    return max(c1, c2, c3)


def _max_contain(c, a_deg, R, oc):
    worst = 0.0
    for i in range(len(c)):
        verts = _hex_vertices(c[i, 0], c[i, 1], 1.0, a_deg[i])
        for v in verts:
            q = v - oc
            viol = _contain_violation(q, R)
            if viol > worst:
                worst = viol
    return worst


def _objective(x):
    R = x[0]
    c = x[1:1 + 3 * 11].reshape(11, 3)
    a_deg = c[:, 2]
    cc = c[:, :2]
    a = np.deg2rad(a_deg)
    contain = _max_contain(cc, a_deg, R, np.array([0.0, 0.0]))
    min_gap = _min_pair_gap(cc, a)
    pen = 80.0
    obj = R + pen * max(0.0, -min_gap) + pen * max(0.0, contain)
    return obj


def _objective_grad(x):
    eps = 1e-5
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (_objective(xp) - _objective(xm)) / (2.0 * eps)
    return g


def _to_inner(x):
    return x[1:1 + 3 * 11].reshape(11, 3)


def _valid(inner, outer, side):
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    c = inner[:, :2]
    a_deg = inner[:, 2]
    a = np.deg2rad(a_deg)
    for i in range(11):
        for j in range(i + 1, 11):
            if _pair_gap(c[i], a[i], c[j], a[j]) <= 1e-6:
                return False
    oc = outer[:2]
    for i in range(11):
        verts = _hex_vertices(c[i, 0], c[i, 1], 1.0, a_deg[i])
        for v in verts:
            if _contain_violation(v - oc, side) > 1e-6:
                return False
    for i in range(11):
        verts = _hex_vertices(c[i, 0], c[i, 1], 1.0, a_deg[i])
        for k in range(6):
            L = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(L - 1.0) > 1e-8:
                return False
    return True


def _hex_grid_init():
    pts = [[0.0, 0.0]]
    ring = [[2, 0], [1, np.sqrt(3)], [-1, np.sqrt(3)],
            [-2, 0], [-1, -np.sqrt(3)], [1, -np.sqrt(3)]]
    for p in ring:
        pts.append([p[0], p[1]])
    extra = [[3, np.sqrt(3) * 1.5], [-3, np.sqrt(3) * 1.5],
             [3, -np.sqrt(3) * 1.5], [-3, -np.sqrt(3) * 1.5]]
    for p in extra:
        pts.append([p[0], p[1]])
    pts = np.array(pts)
    worst = 0.0
    for p in pts:
        worst = max(worst, _contain_violation(p, 1.0))
    s0 = worst + 1.0 + 0.5
    x = np.zeros(1 + 3 * 11)
    x[0] = s0
    for i in range(11):
        x[1 + 3 * i] = pts[i, 0]
        x[1 + 3 * i + 1] = pts[i, 1]
        x[1 + 3 * i + 2] = 0.0
    return x


def _optimize(x0, max_iter):
    return minimize(_objective, x0, method='L-BFGS-B',
                    jac=_objective_grad,
                    options={'maxiter': max_iter, 'maxfun': max_iter * 40,
                             'gtol': 1e-9, 'ftol': 1e-13})


def hexagon_packing_11():
    deadline = _T0 + 600.0
    x0 = _hex_grid_init()
    best_x = x0.copy()
    best_obj = _objective(x0)

    # multi-start with small perturbations
    rng = np.random.default_rng(0)
    starts = [x0]
    for k in range(4):
        xp = x0.copy()
        xp[1:1 + 3 * 11] += rng.normal(0, 0.05, size=(3 * 11,))
        # keep R same
        starts.append(xp)

    for s in starts:
        if time.time() > deadline - 60:
            break
        res = _optimize(s, 150)
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
        # refine
        res2 = _optimize(res.x, 250)
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_x = res2.x.copy()
        if time.time() > deadline - 30:
            break

    inner = _to_inner(best_x)
    side = float(best_x[0])

    if not _valid(inner, np.array([0.0, 0.0, 0.0]), side):
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        side = 8.0

    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


# EVOLVE-BLOCK-END

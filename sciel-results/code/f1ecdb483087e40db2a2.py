"""Packing solver: minimize enclosing regular-hexagon side length for 12 unit hexagons."""
import time
import math
import numpy as np

N = 12
L = 2.0                       # circumradius of a unit regular hexagon
S3 = math.sqrt(3.0)

# vertices of a unit regular hexagon (circumradius 1) at angle 0
_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])


def _hex_vertices(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return np.column_stack((cx + L * (R @ _VERTS.T)[0], cy + L * (R @ _VERTS.T)[1]))


def _enclosing_side(points, outer_deg):
    """Minimum side length of a regular hexagon (orientation outer_deg) containing points."""
    a = math.radians(outer_deg)
    n = np.array([math.cos(a + math.radians(k * 60.0)) for k in range(6)])
    m = np.array([math.sin(a + math.radians(k * 60.0)) for k in range(6)])
    sup = np.max(points * np.column_stack((n, m)), axis=1)   # support in each normal dir
    return float(np.max(sup) / (S3 * L))


def _objective(params):
    pts = np.empty((N * 6, 2))
    for i in range(N):
        pts[i * 6:(i + 1) * 6] = _hex_vertices(params[3 * i], params[3 * i + 1], params[3 * i + 2])
    return _enclosing_side(pts, 0.0)


def _initial_params():
    cells = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    extra = [(2, 0), (2, -1), (-2, 0), (2, -2), (-2, 2)]
    cells = cells + extra
    cx = np.array([2.0 * (a + b / 2.0) for a, b in cells])
    cy = np.array([S3 * b for a, b in cells])
    cx -= cx.mean(); cy -= cy.mean()
    ang = np.zeros(N)
    return np.concatenate([cx, cy, ang])


def _local_polish(params, deadline, min_time=2.0):
    """Nelder-Mead local refinement with deadline guard."""
    from scipy.optimize import minimize
    best = params.copy()
    bestf = _objective(best)
    while time.time() < deadline - min_time:
        try:
            res = minimize(_objective, best, method="Nelder-Mead",
                           options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-9,
                                    "adaptive": True, "initial_simplex_size": 0.4})
        except Exception:
            break
        f = _objective(res.x)
        if f < bestf - 1e-12:
            best, bestf = res.x.copy(), f
        else:
            break
    return best, bestf


def _rotate_params(params, theta_deg):
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    out = params.copy()
    for i in range(N):
        x, y = params[3 * i], params[3 * i + 1]
        out[3 * i] = c * x - s * y
        out[3 * i + 1] = s * x + c * y
        out[3 * i + 2] = params[3 * i + 2] + theta_deg
    return out


def _outer_side(params, theta_deg):
    pts = np.empty((N * 6, 2))
    for i in range(N):
        pts[i * 6:(i + 1) * 6] = _hex_vertices(params[3 * i], params[3 * i + 1], params[3 * i + 2])
    return _enclosing_side(pts, theta_deg)


def _global_rotation(params, deadline):
    """Find the global rotation theta minimizing the enclosing side length."""
    n_scan = 1800
    thetas = np.linspace(0.0, 60.0, n_scan, endpoint=False)
    vals = np.array([_outer_side(params, float(t)) for t in thetas])
    i = int(np.argmin(vals))
    t0 = thetas[i - 1] if i > 0 else thetas[0]
    t1 = thetas[(i + 1) % n_scan]
    # refine around the best scan point
    for _ in range(3):
        lo, hi = max(0.0, t0 - 0.5), min(60.0, t1 + 0.5)
        n = 60
        ts = np.linspace(lo, hi, n)
        vs = np.array([_outer_side(params, float(t)) for t in ts])
        j = int(np.argmin(vs))
        t0, t1 = ts[max(0, j - 1)], ts[min(n - 1, j + 1)]
        if time.time() > deadline - 1.0:
            break
    ts = np.linspace(t0, t1, 200)
    vs = np.array([_outer_side(params, float(t)) for t in ts])
    best_t = float(ts[int(np.argmin(vs))])
    best_v = float(np.min(vs))
    # final polish of theta
    for _ in range(4):
        lo, hi = max(0.0, best_t - 0.05), min(60.0, best_t + 0.05)
        ts = np.linspace(lo, hi, 100)
        vs = np.array([_outer_side(params, float(t)) for t in ts])
        best_t = float(ts[int(np.argmin(vs))])
        best_v = float(np.min(vs))
    return best_t, best_v


def construct():
    t_start = time.time()
    deadline = t_start + 145.0

    params = _initial_params()

    # local polish from the initial placement
    params, f0 = _local_polish(params, deadline)

    # global rotation search
    theta, side = _global_rotation(params, deadline)

    # final local polish in the rotated frame, then re-search rotation
    if time.time() < deadline - 3.0:
        rp = _rotate_params(params, theta)
        rp, _ = _local_polish(rp, deadline)
        theta2, side2 = _global_rotation(rp, deadline)
        if side2 < side:
            params, theta, side = rp, theta2, side2

    # build the inner list: rotate each hexagon by theta (about origin) and report
    inner = []
    for i in range(N):
        x, y, ang = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        inner.append([float(x), float(y), float(ang + theta)])

    return inner, (0.0, 0.0), float(side), float(theta)

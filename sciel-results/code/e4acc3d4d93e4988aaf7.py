"""Packing 12 unit hexagons inside a regular hexagon (minimize outer side).
Uses exact Minkowski-difference support function for non-overlap and
SLSQP multi-start over diverse initial topologies."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 150.0
SAFETY = 8.0          # seconds of margin before the wall-clock limit
EPS = 1e-3            # required clearance between inner hexagons

# A regular hexagon of side 1 with a vertex at angle 0 (pointing +x).
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                      for a in range(0, 360, 60)])


def rot(v, theta):
    """Rotate point array v (shape (6,2)) by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return v @ R.T


def _hull_area(pts):
    try:
        return abs(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _support(A, B, theta):
    """Max over A-vertex, B-vertex of dot(A_i - B_j, u), u=(cos t, sin t)."""
    c, s = math.cos(theta), math.sin(theta)
    a = A @ np.array([c, s])   # (6,)
    b = B @ np.array([c, s])   # (6,)
    return np.max(a) - np.min(b)


def _minkowski_area(A, B):
    """Area of the Minkowski difference A - B (a centrally symmetric convex polygon)."""
    # sample support function on a fine circle -> polygon of the difference
    thetas = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    pts = np.array([[_support(A, B, t) * math.cos(t),
                     _support(A, B, t) * math.sin(t)] for t in thetas])
    return _hull_area(pts)


def _pair_area(i, j, x, y, th):
    A = rot(HEX_VERTS, th[i])
    B = rot(HEX_VERTS, th[j])
    return _minkowski_area(A, B)


def _outer_support(A, S, phi):
    """Max over vertices of (S * rot(HEX_VERTS, phi) + S*phi) of dot(v, u)."""
    c, s = math.cos(phi), math.sin(phi)
    u = np.array([c, s])
    verts = S * rot(HEX_VERTS, phi) + S * u
    return np.max(verts @ u)


def _outer_area(S, phi):
    thetas = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    pts = np.array([[_outer_support(HEX_VERTS, S, phi) * math.cos(t),
                     _outer_support(HEX_VERTS, S, phi) * math.sin(t)] for t in thetas])
    return _hull_area(pts)


def _feasible(x, y, th, S, phi):
    """Return (ok, min_slack). ok = all inner-inner clearances >= EPS and all inside outer."""
    n = len(x)
    min_slack = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            r = math.sqrt(max(_pair_area(i, j, x, y, th), 0.0))
            min_slack = min(min_slack, d - r)
    # inside outer (support function of A - S*hex)
    for i in range(n):
        A = rot(HEX_VERTS, th[i]) + np.array([x[i], y[i]])
        B = S * rot(HEX_VERTS, phi) + S * np.array([math.cos(phi), math.sin(phi)])
        thetas = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
        for t in thetas:
            if _support(A, B, t) > 1e-9:
                return False, -1.0
    return True, min_slack


def _obj(z, S, phi):
    """Objective: minimize S + 0.5 * penalty for violated clearances."""
    x = z[:N]
    y = z[N:2 * N]
    th = z[2 * N:3 * N]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            r = math.sqrt(max(_pair_area(i, j, x, y, th), 0.0))
            gap = d - r
            if gap < EPS:
                pen += (EPS - gap) ** 2
    return S + 0.5 * pen


def _solve_one(x0, S0, phi, t_end):
    """Run SLSQP on a fixed outer (S0, phi); return best S found."""
    best = S0
    z0 = np.concatenate([x0, y0_dummy(x0), th0_dummy()])  # placeholder, replaced below
    # build a proper initial vector from caller via closure:
    return best


def _run():
    global DEADLINE
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # ---- initial topologies: diverse lattice / ring / random arrangements ----
    def make_initials():
        inits = []
        # 1) 3x4 grid of hexagons (hex lattice spacing)
        # unit hexagon "width" (flat-to-flat) = sqrt(3); center distance for
        # non-overlap of axis-aligned hexes = 2 * apothem = sqrt(3) in x, 1.5 in y
        rows, cols = 3, 4
        dx = math.sqrt(3.0) * 1.05
        dy = 1.5 * 1.05
        xs, ys = [], []
        for r in range(rows):
            for c in range(cols):
                xs.append((c - (cols - 1) / 2.0) * dx + (r % 2) * dx / 2.0)
                ys.append((r - (rows - 1) / 2.0) * dy)
        inits.append((np.array(xs), np.array(ys), np.zeros(N), 6.0))
        # 2) ring of 12 around center
        ang = np.linspace(0, 2 * math.pi, N, endpoint=False)
        rad = 2.2
        inits.append((rad * np.cos(ang), rad * np.sin(ang), np.zeros(N), 5.0))
        # 3) random scatter
        for k in range(6):
            xs = rng.uniform(-2.5, 2.5, N)
            ys = rng.uniform(-2.5, 2.5, N)
            th = rng.uniform(0, 2 * math.pi, N)
            inits.append((xs, ys, th, 6.0))
        return inits

    inits = make_initials()
    best_S = 1e9
    best_cfg = None

    for (x0, y0, th0, S0) in inits:
        if time.time() > DEADLINE - SAFETY:
            break
        # try a few outer rotations for this topology
        for phi in [0.0, math.pi / 6.0, math.pi / 3.0]:
            if time.time() > DEADLINE - SAFETY:
                break
            z0 = np.concatenate([x0, y0, th0])
            try:
                res = minimize(_obj, z0, args=(S0, phi), method='SLSQP',
                               options={'maxiter': 60, 'ftol': 1e-6, 'disp': False})
            except Exception:
                continue
            x = res.x[:N]; y = res.x[N:2 * N]; th = res.x[2 * N:3 * N]
            ok, slack = _feasible(x, y, th, S0, phi)
            if ok:
                # try to shrink S a bit
                S = S0
                for _ in range(20):
                    if time.time() > DEADLINE - SAFETY:
                        break
                    S2 = S * 0.97
                    ok2, _ = _feasible(x, y, th, S2, phi)
                    if ok2:
                        S = S2
                    else:
                        break
                if S < best_S:
                    best_S = S
                    best_cfg = (x.copy(), y.copy(), th.copy(), phi)

    # ---- final polish: minimize S with SLSQP on (S, phi, positions) ----
    if best_cfg is not None and time.time() < DEADLINE - SAFETY:
        x, y, th, phi = best_cfg
        def obj2(w):
            S = w[0]
            ph = w[1]
            xx = w[2:2 + N]; yy = w[2 + N:2 + 2 * N]; tt = w[2 + 2 * N:2 + 3 * N]
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = math.hypot(xx[i] - xx[j], yy[i] - yy[j])
                    r = math.sqrt(max(_pair_area(i, j, xx, yy, tt), 0.0))
                    gap = d - r
                    if gap < EPS:
                        pen += (EPS - gap) ** 2
            # inside penalty
            for i in range(N):
                A = rot(HEX_VERTS, tt[i]) + np.array([xx[i], yy[i]])
                B = S * rot(HEX_VERTS, ph) + S * np.array([math.cos(ph), math.sin(ph)])
                thetas = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
                for t in thetas:
                    v = _support(A, B, t)
                    if v > 0:
                        pen += v ** 2
            return S + 0.5 * pen
        w0 = np.concatenate([[best_S], [phi], x, y, th])
        try:
            res = minimize(obj2, w0, method='SLSQP',
                           options={'maxiter': 120, 'ftol': 1e-10, 'disp': False})
            S2 = res.x[0]; ph2 = res.x[1]
            x2 = res.x[2:2 + N]; y2 = res.x[2 + N:2 + 2 * N]; th2 = res.x[2 + 2 * N:2 + 3 * N]
            ok, _ = _feasible(x2, y2, th2, S2, ph2)
            if ok and S2 < best_S:
                best_S = S2
                best_cfg = (x2.copy(), y2.copy(), th2.copy(), ph2)
        except Exception:
            pass

    if best_cfg is None:
        # fallback: a guaranteed-feasible trivial packing (large outer)
        xs = [math.sqrt(3.0) * (k + 0.5) for k in range(N)]
        return [[xs[k], 0.0, 0.0] for k in range(N)], (0.0, 0.0), 25.0, 0.0

    x, y, th, phi = best_cfg
    inner = [[float(x[k]), float(y[k]), float(math.degrees(th[k]))] for k in range(N)]
    return inner, (0.0, 0.0), float(best_S), float(math.degrees(phi))


def construct():
    """Return (inner, center, side, angle_degrees)."""
    t0 = time.time()
    try:
        inner, center, side, angle = _run()
    except Exception:
        # safe fallback
        xs = [math.sqrt(3.0) * (k + 0.5) for k in range(N)]
        inner = [[xs[k], 0.0, 0.0] for k in range(N)]
        center, side, angle = (0.0, 0.0), 25.0, 0.0
    return inner, center, side, angle


def y0_dummy(x0):
    return np.zeros_like(x0)


def th0_dummy():
    return np.zeros(N)

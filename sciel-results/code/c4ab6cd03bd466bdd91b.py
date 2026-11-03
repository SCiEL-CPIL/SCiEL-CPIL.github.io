import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 140.0          # safety margin under the 150 s wall-clock kill
_t0 = time.time()
_LAMBDA = 500.0           # overlap penalty weight


def _hex_vertices(cx, cy, ang_deg, side=1.0):
    a = np.deg2rad(ang_deg)
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + side * np.cos(th), cy + side * np.sin(th)])


def _all_vertices(inner):
    pts = [_hex_vertices(x, y, ang) for (x, y, ang) in inner]
    return np.vstack(pts)


def _enclosing_fast(V, theta_cands):
    """Fast upper bound on the minimal enclosing-hexagon side (c = centroid)."""
    c = V.mean(axis=0)
    best = np.inf
    for th in theta_cands:
        a = np.deg2rad(th)
        normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3.0),
                                   np.sin(a + np.arange(6) * np.pi / 3.0)])
        proj = V @ normals.T
        m = proj.max(axis=0) - normals @ c
        s = (2.0 / np.sqrt(3.0)) * m.max()
        if s < best:
            best = s
    return best


def _enclosing_full(V):
    """Exact minimal enclosing-hexagon side; optimize (cx, cy, theta)."""
    def obj(p):
        cx, cy, th = p
        a = np.deg2rad(th)
        normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3.0),
                                   np.sin(a + np.arange(6) * np.pi / 3.0)])
        proj = V @ normals.T
        m = proj.max(axis=0) - normals @ np.array([cx, cy])
        return (2.0 / np.sqrt(3.0)) * m.max()
    c0 = V.mean(axis=0)
    res = minimize(obj, [c0[0], c0[1], 0.0], method='Nelder-Mead',
                   options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-9})
    return float(res.fun), (float(res.x[0]), float(res.x[1])), float(res.x[2])


def _sat_penetration(A, B):
    """Minimum translation depth if convex hexagons A,B overlap, else 0."""
    axes = []
    for P in (A, B):
        for i in range(6):
            e = P[(i + 1) % 6] - P[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    min_pen = np.inf
    for ax in axes:
        pa = A @ ax
        pb = B @ ax
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if overlap <= 0.0:
            return 0.0
        if overlap < min_pen:
            min_pen = overlap
    return min_pen


def _total_penetration(inner):
    verts = [_hex_vertices(x, y, ang) for (x, y, ang) in inner]
    tot = 0.0
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            tot += _sat_penetration(verts[i], verts[j])
    return tot


def _initial_inner():
    """Exact pointy-top honeycomb tiling of 12 unit hexagons (no overlap)."""
    d = np.sqrt(3.0)
    pos = []
    for row in range(3):
        for col in range(4):
            x = col * d + (0.5 * d if row % 2 else 0.0)
            y = row * 1.5
            pos.append((x, y))
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    mx = (min(xs) + max(xs)) / 2.0
    my = (min(ys) + max(ys)) / 2.0
    pos = [(x - mx, y - my) for (x, y) in pos]
    return [[x, y, 30.0] for (x, y) in pos]


def _flatten(inner):
    return [v for row in inner for v in row]


def _unflatten(flat):
    return [[flat[i], flat[i + 1], flat[i + 2]] for i in range(0, len(flat), 3)]


def construct():
    global _t0
    theta_cands = np.linspace(0.0, 60.0, 13)[:-1]
    x0 = _flatten(_initial_inner())
    bounds = []
    for _ in range(N):
        bounds += [(-12.0, 12.0), (-12.0, 12.0), (-180.0, 180.0)]

    def de_obj(flat):
        if time.time() - _t0 > DEADLINE:
            return 1e9
        inner = _unflatten(flat)
        V = _all_vertices(inner)
        s = _enclosing_fast(V, theta_cands)
        pen = _total_penetration(inner)
        return s + _LAMBDA * pen

    best_flat = None
    try:
        res = differential_evolution(de_obj, bounds, seed=42, maxiter=70,
                                     popsize=22, tol=1e-5, polish=False,
                                     init='sobol')
        if res.x is not None and res.fun < 1e8:
            best_flat = list(res.x)
    except Exception:
        best_flat = None

    if best_flat is None:
        best_flat = list(x0)

    # Polish the outer hexagon exactly for the best inner configurations.
    candidates = [best_flat, list(x0)]
    best_overall = None
    for cand in candidates:
        if time.time() - _t0 > DEADLINE - 5.0:
            break
        inner = _unflatten(cand)
        V = _all_vertices(inner)
        try:
            s, center, th = _enclosing_full(V)
        except Exception:
            continue
        if best_overall is None or s < best_overall[0]:
            best_overall = (s, inner, center, th)

    if best_overall is None:
        inner = _unflatten(best_flat)
        V = _all_vertices(inner)
        s = _enclosing_fast(V, theta_cands)
        c = V.mean(axis=0)
        return inner, (float(c[0]), float(c[1])), float(s), 0.0

    s, inner, center, th = best_overall
    return inner, center, float(s), float(th)

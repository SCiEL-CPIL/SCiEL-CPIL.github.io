"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 143.0  # safety margin before 150s kill
EPS = 1e-9


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(circles):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < 0) or np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    n = len(circles)
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[iu[0]] + r[iu[1]] - 1e-9):
        return False
    return True


def _grow_all(circles):
    """Grow each circle to its max feasible radius given current positions."""
    n = len(circles)
    x, y, r = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
    for i in range(n):
        max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 0:
                max_r = min(max_r, dist - r[j])
        r[i] = max(EPS, max_r)
    circles[:, 2] = r
    return circles


def _penalty_objective(params, penalty=5000.0):
    p = params.reshape(N, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    score = -np.sum(r)
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    return score + penalty * (np.sum(overlap) + wall)


def _penalty_grad(params, penalty=5000.0):
    p = params.reshape(N, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    g = np.zeros_like(params)
    g[2::3] = -1.0
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0); g[3*i+2] += penalty
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty; g[3*i+2] += penalty
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0); g[3*i+2] += penalty
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty; g[3*i+2] += penalty
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    ds = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]; ay = iu[1][active]
        adx = dx[active] / ds[active]; ady = dy[active] / ds[active]
        for k in range(len(ax)):
            i = ax[k]; j = ay[k]
            g[3*i] += penalty * (-adx[k]); g[3*j] += penalty * (adx[k])
            g[3*i+1] += penalty * (-ady[k]); g[3*j+1] += penalty * (ady[k])
            g[3*i+2] += penalty; g[3*j+2] += penalty
    return g


def _enforce_validity(params):
    """Shrink radii / clamp to make a configuration valid (never increases score)."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], EPS, 0.5)
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, EPS))
    for _ in range(120):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]; j = iu[1][k]; ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                s = min(ov, p[i, 2] - EPS); p[i, 2] -= s
                rem = ov - s
                if rem > 0:
                    s2 = min(rem, p[j, 2] - EPS); p[j, 2] -= s2
            else:
                s = min(ov, p[j, 2] - EPS); p[j, 2] -= s
                rem = ov - s
                if rem > 0:
                    s2 = min(rem, p[i, 2] - EPS); p[i, 2] -= s2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, EPS))
    return p


def _delaunay_edges(circles):
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for s in tri.simplices:
            for k in range(3):
                a, b = s[k], s[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
    except Exception:
        edges = set()
        for i in range(N):
            for j in range(i + 1, N):
                edges.add((i, j))
    # Ensure active overlap pairs are included
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] + 1e-6:
                edges.add((min(i, j), max(i, j)))
    return list(edges)


def _force_polish(circles, max_time=8.0):
    """Local force relaxation on the Delaunay contact graph to tighten packing."""
    start = time.time()
    n = len(circles)
    best = circles.copy()
    best_val = np.sum(best[:, 2])
    lr = 0.05
    for _ in range(4000):
        if time.time() - start > max_time:
            break
        if _remaining() < 4:
            break
        circles = _grow_all(circles)
        x, y, r = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
        fx = np.zeros(n); fy = np.zeros(n)
        for (i, j) in _delaunay_edges(circles):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-10:
                continue
            gap = d - r[i] - r[j]
            if gap < 0:
                f = -gap
                fx[i] += f * dx / d; fy[i] += f * dy / d
                fx[j] -= f * dx / d; fy[j] -= f * dy / d
        x = np.clip(x + lr * fx, 0.0, 1.0)
        y = np.clip(y + lr * fy, 0.0, 1.0)
        circles[:, 0] = x; circles[:, 1] = y
        circles = _grow_all(circles)
        val = np.sum(circles[:, 2])
        if val > best_val:
            best = circles.copy(); best_val = val
        lr *= 0.995
        if lr < 1e-4:
            lr = 1e-4
    return best


def _lbfgs(params, max_time=12.0, penalty=5000.0):
    """Run L-BFGS-B on the penalty objective; return best valid result."""
    start = time.time()
    lo = np.zeros(3 * N); hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0; hi[3*i] = 1.0
        lo[3*i+1] = 0.0; hi[3*i+1] = 1.0
        lo[3*i+2] = EPS; hi[3*i+2] = 0.5
    maxiter = 200
    res = minimize(_penalty_objective, params, jac=_penalty_grad,
                   method='L-BFGS-B', bounds=list(zip(lo, hi)),
                   options={'maxiter': maxiter, 'ftol': 1e-12,
                            'maxfun': 4000, 'maxls': 40})
    if time.time() - start > max_time:
        return None
    out = None
    for cand in (res.x, params):
        c = _grow_all(_enforce_validity(cand.reshape(N, 3)))
        if _validate(c):
            if out is None or np.sum(c[:, 2]) > np.sum(out[:, 2]):
                out = c
    return out


def _shake_refine(circles, time_limit=6.0, seed=0):
    """Shake positions, re-grow, keep improvements (escape local minima)."""
    rng = np.random.RandomState(seed)
    start = time.time()
    best = circles.copy()
    best_val = np.sum(best[:, 2])
    sigma = 0.02
    for _ in range(2000):
        if time.time() - start > time_limit or _remaining() < 4:
            break
        c = best.copy()
        c[:, :2] += rng.normal(0, sigma, size=(N, 2))
        c[:, :2] = np.clip(c[:, :2], 0.0, 1.0)
        c = _grow_all(c)
        if _validate(c):
            val = np.sum(c[:, 2])
            if val > best_val:
                best = c; best_val = val
    return best


def _hex_initial(seed):
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            pts.append((x, y))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    scale = 0.45 / np.max(np.abs(pts))
    pts *= scale
    pts += 0.5
    d = np.sum((pts - 0.5) ** 2, axis=1)
    idx = np.argsort(d)[:N]
    p = pts[idx]
    c = np.empty((N, 3))
    c[:, 0] = p[:, 0]; c[:, 1] = p[:, 1]; c[:, 2] = 0.05
    c[:, :2] += rng.uniform(-1e-4, 1e-4, size=(N, 2))
    return _grow_all(_enforce_validity(c))


def _grid_initial(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    c = np.array(cells[:N], dtype=float)
    c[:, :2] += rng.uniform(-1e-5, 1e-5, size=(N, 2))
    return _grow_all(_enforce_validity(c))


def _voronoi_initial(seed):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.05, 0.95, size=(N, 2))
    r = np.zeros(N)
    for i in range(N):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d[i] = np.inf
        r[i] = 0.5 * np.min([np.min(d), pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1]]) * 0.95
    c = np.empty((N, 3))
    c[:, 0] = pts[:, 0]; c[:, 1] = pts[:, 1]; c[:, 2] = r
    c += rng.uniform(-1e-5, 1e-5, size=c.shape)
    return _grow_all(_enforce_validity(c))


def _jitter_initial(seed):
    rng = np.random.RandomState(seed)
    c = np.empty((N, 3))
    c[:, 0] = rng.uniform(0.1, 0.9, size=N)
    c[:, 1] = rng.uniform(0.1, 0.9, size=N)
    c[:, 2] = 0.03
    return _grow_all(_enforce_validity(c))


def construct():
    global _start_time
    _start_time = time.time()

    best = None
    best_val = -np.inf
    inits = [_hex_initial, _grid_initial, _voronoi_initial, _jitter_initial]
    seeds = [0, 1, 2, 3, 4, 5]

    for s in seeds:
        if _remaining() < 20:
            break
        for mk in inits:
            if _remaining() < 20:
                break
            c = mk(s)
            val = np.sum(c[:, 2])
            if val > best_val:
                best, best_val = c, val

    # Gradient refinement on the best candidate
    if best is not None and _remaining() > 25:
        c = _lbfgs(best.flatten(), max_time=min(12.0, _remaining() - 6), penalty=5000.0)
        if c is not None:
            c = _force_polish(c, max_time=min(6.0, _remaining() - 6))
            c = _shake_refine(c, time_limit=min(5.0, _remaining() - 4), seed=7)
            if _validate(c):
                val = np.sum(c[:, 2])
                if val > best_val:
                    best, best_val = c, val

    # Second pass from a different seed if time remains
    if _remaining() > 30:
        c = _hex_initial(11)
        c = _lbfgs(c.flatten(), max_time=min(14.0, _remaining() - 8), penalty=8000.0)
        if c is not None:
            c = _force_polish(c, max_time=min(6.0, _remaining() - 6))
            c = _shake_refine(c, time_limit=min(5.0, _remaining() - 4), seed=13)
            if _validate(c):
                val = np.sum(c[:, 2])
                if val > best_val:
                    best, best_val = c, val

    # Final polish on the best
    if best is not None and _remaining() > 6:
        c = _force_polish(best, max_time=min(5.0, _remaining() - 3))
        if _validate(c) and np.sum(c[:, 2]) > best_val:
            best = c

    if best is None or not _validate(best):
        best = _grid_initial(0)

    return best
# EVOLVE-BLOCK-END

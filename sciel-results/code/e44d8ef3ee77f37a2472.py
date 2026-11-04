"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 26
DEADLINE = 115.0  # safety margin before 120s kill


def _smooth_overlap(p, r, k=60.0):
    """Smooth penalty for overlap + boundary violations. Returns (value, n_active)."""
    n = len(p)
    pen = 0.0
    cnt = 0
    # pairwise
    for i in range(n):
        xi, yi, ri = p[i]
        for j in range(i + 1, n):
            xj, yj, rj = p[j]
            d = np.hypot(xi - xj, yi - yj)
            s = ri + rj - d
            if s > -0.05:
                cnt += 1
                pen += np.log(1.0 + np.exp(k * s)) / k
    # walls
    for i in range(n):
        x, y, r = p[i]
        for s in (r - x, r - y, r - (1.0 - x), r - (1.0 - y)):
            if s > -0.05:
                cnt += 1
                pen += np.log(1.0 + np.exp(k * s)) / k
    return pen, cnt


def _objective(v, k=60.0):
    """Objective = -sum(r) + penalty. Lower is better."""
    n = N
    p = v.reshape(n, 3)
    r = p[:, 2]
    pen, _ = _smooth_overlap(p, r, k)
    return -np.sum(r) + pen


def _is_valid(p, tol=1e-6):
    n = len(p)
    for i in range(n):
        x, y, r = p[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1.0 + tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
            if d < p[i, 2] + p[j, 2] - tol:
                return False
    return True


def _sum_r(p):
    return float(np.sum(p[:, 2]))


def _seed():
    """Geometric seed: rings of circles, radii decreasing with ring index."""
    n = N
    pts = []
    # center
    pts.append([0.5, 0.5, 0.28])
    # ring 1: 6 circles
    for a in range(6):
        th = a * 2 * np.pi / 6
        pts.append([0.5 + 0.30 * np.cos(th), 0.5 + 0.30 * np.sin(th), 0.16])
    # ring 2: 8 circles
    for a in range(8):
        th = a * 2 * np.pi / 8
        pts.append([0.5 + 0.42 * np.cos(th), 0.5 + 0.42 * np.sin(th), 0.10])
    # ring 3: remaining
    need = n - len(pts)
    for a in range(need):
        th = a * 2 * np.pi / max(need, 1)
        pts.append([0.5 + 0.48 * np.cos(th), 0.5 + 0.48 * np.sin(th), 0.07])
    return np.array(pts[:n], dtype=float)


def _clamp(v):
    p = v.reshape(N, 3)
    r = np.clip(p[:, 2], 0.001, 0.5)
    p[:, 2] = r
    p[:, 0] = np.clip(p[:, 0], r, 1.0 - r)
    p[:, 1] = np.clip(p[:, 1], r, 1.0 - r)
    return p


def _refine(x0, k, iters, deadline, label):
    """Gradient refinement with adaptive smoothing. Returns best p."""
    best = x0.copy()
    best_obj = _objective(best, k)
    x = x0.copy()
    for it in range(iters):
        if time.time() > deadline:
            break
        res = minimize(_objective, x, args=(k,), method='L-BFGS-B',
                       options={'maxiter': 60, 'ftol': 1e-9})
        x = res.x
        p = _clamp(x)
        obj = _objective(x, k)
        if obj < best_obj:
            best_obj = obj
            best = x.copy()
        # adapt k
        pen, cnt = _smooth_overlap(_clamp(x), _clamp(x)[:, 2], k)
        if cnt == 0 and pen < 1e-4:
            k = min(k * 1.3, 400.0)
        elif pen > 1.0:
            k = max(k * 0.7, 20.0)
    return _clamp(best)


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best = None
    best_val = -np.inf

    def consider(p, tag):
        nonlocal best, best_val
        if not _is_valid(p):
            return
        v = _sum_r(p)
        if v > best_val:
            best_val = v
            best = p.copy()

    # Stage 1: multi-start gradient refinement from several seeds
    seeds = [_seed()]
    # grid-based seed
    side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append([0.5 / side + i / side, 0.5 / side + j / side, 0.45 / side])
    seeds.append(np.array(grid[:N], dtype=float))
    # random seed
    rng = np.random.default_rng(0)
    rx = rng.uniform(0.1, 0.9, N)
    ry = rng.uniform(0.1, 0.9, N)
    rr = rng.uniform(0.06, 0.2, N)
    seeds.append(np.column_stack([rx, ry, rr]))

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = _clamp(s.reshape(-1, 3).reshape(-1))
        p = _refine(s, 60.0, 6, deadline, f"seed{si}")
        consider(p, f"refine{si}")

    # Stage 2: global differential evolution (coarse), then refine
    if time.time() < deadline - 20:
        try:
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.4)] * N
            de = differential_evolution(
                _objective, bounds, args=(40.0,),
                maxiter=40, popsize=15, tol=1e-6,
                seed=1, polish=False, init='sobol')
            p = _clamp(de.x)
            consider(p, "de")
            # refine the DE result
            p2 = _refine(de.x, 60.0, 8, deadline, "de_refine")
            consider(p2, "de_refine")
        except Exception:
            pass

    # Stage 3: final polish with high k
    if best is not None and time.time() < deadline - 5:
        p3 = _refine(best.reshape(-1), 150.0, 10, deadline, "polish")
        consider(p3, "polish")

    if best is None:
        # fallback: valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best

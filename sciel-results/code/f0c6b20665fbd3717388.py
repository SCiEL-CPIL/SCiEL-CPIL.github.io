import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _radius(centers):
    c = np.asarray(centers, dtype=float)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5 - 1e-9) or np.any(c > 0.5 + 1e-9):
        return 0.0
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = float(np.min(d)) / 2.0
    wall = 0.5 - np.abs(c)
    r_bound = float(np.min(wall))
    return min(r_pair, r_bound)

def _farthest_sample(points, k, rng):
    M = len(points)
    chosen = [int(rng.integers(M))]
    min_d = np.full(M, np.inf)
    for _ in range(k - 1):
        last = points[chosen[-1]]
        d = np.linalg.norm(points - last, axis=1)
        min_d = np.minimum(min_d, d)
        for cc in chosen:
            min_d[cc] = -1.0
        idx = int(np.argmax(min_d))
        chosen.append(idx)
    return points[np.array(chosen)]

def _make_init(kind, rng):
    if kind == 0:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks], dtype=float)
        grid = grid + rng.normal(0, 0.03, size=grid.shape)
        grid = np.clip(grid, -0.49, 0.49)
        return _farthest_sample(grid, N, rng).copy()
    elif kind == 1:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks], dtype=float)
        grid = grid + rng.normal(0, 0.02, size=grid.shape)
        grid = np.clip(grid, -0.49, 0.49)
        return _farthest_sample(grid, N, rng).copy()
    else:
        c = rng.uniform(-0.45, 0.45, size=(N, 3))
        return c

def _repulsion_opt(centers, bounds, p, maxiter):
    eps = 1e-7
    def obj(x):
        c = x.reshape(N, 3)
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        d = np.maximum(d, eps)
        E = np.sum(d ** (-p))
        wall = np.maximum(0.5 - np.abs(c), eps)
        E += np.sum(wall ** (-p))
        return float(E)
    def grad(x):
        c = x.reshape(N, 3)
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        d = np.maximum(d, eps)
        W = -p * d ** (-(p + 2))
        np.fill_diagonal(W, 0.0)
        g = W @ diff
        wall = np.maximum(0.5 - np.abs(c), eps)
        g += (p * np.sign(c) * wall ** (-(p + 1)))
        return g.reshape(-1)
    res = minimize(obj, centers.ravel(), jac=grad, method='L-BFGS-B',
                   bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': maxiter * 12})
    return res.x.reshape(N, 3)

def construct():
    t0 = time.time()
    DEADLINE = 112.0
    rng = np.random.default_rng(20240501)
    bounds = [(-0.5 + 1e-4, 0.5 - 1e-4)] * (N * 3)

    best = _make_init(0, rng)
    best_r = _radius(best)

    p = 6.0
    restart = 0
    while time.time() - t0 < DEADLINE:
        kind = restart % 3
        init = _make_init(kind, rng)
        r0 = _radius(init)
        # budget per restart shrinks as we approach deadline
        remaining = DEADLINE - (time.time() - t0)
        maxiter = int(min(400, max(60, remaining * 120)))
        if maxiter < 30:
            break
        try:
            cand = _repulsion_opt(init, bounds, p, maxiter)
        except Exception:
            cand = init
        cand = np.clip(cand, -0.49999, 0.49999)
        r = _radius(cand)
        if r > best_r:
            best = cand.copy()
            best_r = r
        # also consider the init if better
        if r0 > best_r:
            best = init.copy(); best_r = r0
        restart += 1

    return best

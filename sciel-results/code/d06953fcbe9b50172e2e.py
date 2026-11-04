"""Pack 50 equal circles in a unit square, maximize common radius."""
N = 50
DEADLINE = 118.0  # safety margin before the 150s kill

import time
import numpy as np
from scipy.optimize import minimize


def _radius_fast(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r_wall = float(np.min(w))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r_pair = float(np.min(d)) * 0.5
    return min(r_wall, r_pair)


def _soft_min_radius(c, k):
    """Smooth surrogate for the min-radius via log-sum-exp."""
    pts = c.reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    dw = np.stack([
        0.5 - pts[:, 0], 0.5 + pts[:, 0],
        0.5 - pts[:, 1], 0.5 + pts[:, 1],
    ])
    all_d = np.concatenate([dists.ravel(), dw.ravel()])
    all_d = np.clip(all_d, 1e-8, 1.0)
    weights = np.exp(-k * all_d)
    return -np.log(weights.sum()) / k


def _relax_hard_disk(c, r, time_budget=2.0, max_iter=5000):
    """Push overlapping circles apart and keep them inside the square."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.03
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = np.where(mask, (2 * r - d) / (2 * r), 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis in range(2):
            pen_r = np.clip(c[:, axis] - (0.5 - r), 0, None)
            disp[:, axis] += pen_r / (2 * r)
            pen_l = np.clip((-0.5 + r) - c[:, axis], 0, None)
            disp[:, axis] -= pen_l / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _init_7x7_plus():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_hexagonal():
    m = 7
    pts = []
    dy = 0.866 / m
    for i in range(m):
        offset = 0.5 / m if i % 2 == 0 else 0.0
        for j in range(m):
            x = (j + offset) / m - 0.5
            y = (i + 0.5) * dy - 0.5 + (0.5 - m * dy / 2)
            pts.append((x, y))
    g = np.array(pts[:N])
    return g


def _init_jittered():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.03, 0.03, g.shape)
    g = g + jitter
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_random(seed=12345):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.42, 0.42, (N, 2))


def _slsqp_epigraph_polish(c0, r_init, time_budget=8.0):
    """SLSQP with epigraph formulation: maximize t subject to constraints."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    # Pre-compute constraint indices for speed
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def pair_con_fun(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        vals = []
        for i, j in pairs:
            d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
            vals.append(d / 2.0 - t)
        return vals

    def pair_con_grad(x):
        pts = x[:2 * N].reshape(N, 2)
        g = np.zeros((len(pairs), 2 * N + 1))
        for idx, (i, j) in enumerate(pairs):
            diff = pts[i] - pts[j]
            d = np.sqrt(np.sum(diff ** 2))
            if d > 1e-12:
                g[idx, 2 * i:2 * i + 2] = diff / (2 * d)
                g[idx, 2 * j:2 * j + 2] = -diff / (2 * d)
            g[idx, 2 * N] = -1.0
        return g

    def wall_con_fun(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        vals = []
        for i in range(N):
            for axis in range(2):
                vals.append(0.5 - x[2 * i + axis] - t)
                vals.append(x[2 * i + axis] + 0.5 - t)
        return vals

    def wall_con_grad(x):
        g = np.zeros((4 * N, 2 * N + 1))
        for i in range(N):
            for axis in range(2):
                g[4 * i, 2 * i + axis] = -1.0
                g[4 * i, 2 * N] = -1.0
                g[4 * i + 1, 2 * i + axis] = 1.0
                g[4 * i + 1, 2 * N] = -1.0
        return g

    cons = [
        {'type': 'ineq', 'fun': pair_con_fun, 'jac': pair_con_grad},
        {'type': 'ineq', 'fun': wall_con_fun, 'jac': wall_con_grad},
    ]

    x0 = np.concatenate([c0.ravel(), [r_init]])

    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                       constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0

    if time.time() - start > time_budget:
        return c
    return c


def _voronoi_style_refine(c0, time_budget=3.0):
    """Move each circle to the centroid of its Voronoi cell (toward neighbors)."""
    c = np.array(c0, dtype=float, copy=True)
    start = time.time()
    step = 0.001

    for _ in range(1000):
        if time.time() - start > time_budget:
            break
        # For each circle, compute direction away from all others
        # and toward the center of its "space"
        disp = np.zeros_like(c)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)

        for i in range(N):
            # Force from walls
            for axis in range(2):
                dist_r = 0.5 - c[i, axis]
                dist_l = c[i, axis] + 0.5
                if dist_r < 2 * _radius_fast(c) * 0.8:
                    disp[i, axis] += 0.5
                if dist_l < 2 * _radius_fast(c) * 0.8:
                    disp[i, axis] -= 0.5

        c = np.clip(c + step * disp, -0.5, 0.5)

    return c


def _final_local_search(c0, time_budget=3.0):
    """Final high-precision polish using soft-min with high k."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x, k):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    k = 200.0
    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       args=(k,),
                       options={'maxiter': 100, 'maxfun': 2000,
                                'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()

    # Generate initial candidates
    inits = [
        _init_7x7_plus(),
        _init_hexagonal(),
        _init_jittered(),
        _init_random(12345),
        _init_random(54321),
    ]

    # Stage 1: Hard-disk radius expansion from several starts
    best = None
    best_r = 0.0

    for c0 in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        r = _radius_fast(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r * 1.5
        cur = c0

        for _ in range(12):
            if time.time() - start_time > DEADLINE - 30:
                break
            mid = 0.5 * (lo + hi)
            relaxed = _relax_hard_disk(cur, mid, time_budget=2.0, max_iter=4000)
            s = _radius_fast(relaxed)
            if s >= mid - 1e-9:
                lo = mid
                cur = relaxed
            else:
                hi = mid
                if hi - lo < 1e-6:
                    break

        s = _radius_fast(cur)
        if s > best_r:
            best_r = s
            best = cur

    # Stage 2: SLSQP epigraph polish from best and other inits
    candidates = [best]
    for c0 in inits[:3]:
        if time.time() - start_time > DEADLINE - 20:
            break
        r_est = _radius_fast(c0) * 0.9
        relaxed = _relax_hard_disk(c0, r_est, time_budget=2.0)
        candidates.append(relaxed)

    for c0 in candidates:
        if time.time() - start_time > DEADLINE - 15:
            break
        cur = _slsqp_epigraph_polish(c0, best_r * 0.98, time_budget=6.0)
        s = _radius_fast(cur)
        if s > best_r:
            best_r = s
            best = cur

    # Stage 3: Soft-min polish with high k
    if time.time() - start_time < DEADLINE - 10:
        c_final = _final_local_search(best, time_budget=5.0)
        s = _radius_fast(c_final)
        if s > best_r:
            best_r = s
            best = c_final

    # Stage 4: One more SLSQP from the best if time permits
    if time.time() - start_time < DEADLINE - 5:
        cur = _slsqp_epigraph_polish(best, best_r * 0.99, time_budget=4.0)
        s = _radius_fast(cur)
        if s > best_r:
            best_r = s
            best = cur

    return best

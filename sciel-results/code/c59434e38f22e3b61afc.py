"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy:
1. Multiple diverse initializations (hexagonal, grid, random).
2. Hard-disk radius expansion with adaptive step sizes.
3. Multi-temperature L-BFGS-B soft-min polish.
4. Active-set refinement on the kissing graph (QCQP-style).
5. Final local search with very small perturbations.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 112.0  # safety margin before 150s kill


def _radius(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_d = d.min()
    return min(wall_r, min_pair_d / 2.0)


def _radius_fast(c):
    """Faster radius using cdist."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius_neg(x, t=50.0):
    """Negative soft-min radius for minimization."""
    c = x.reshape(N, 2)
    # Wall terms
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    # Pairwise terms
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return -soft


def _soft_min_radius_grad(x, t=50.0):
    """Gradient of negative soft-min radius."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()

    grad_flat = np.zeros(2 * N)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad

    # Pair gradients
    # For pair (i,j): term = ||c_i - c_j|| / 2
    # d/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 1e-15:
                dij = c[i] - c[j]
                norm = d[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad_flat[i * 2] += total_w * g[0]
                    grad_flat[i * 2 + 1] += total_w * g[1]
                    grad_flat[j * 2] -= total_w * g[0]
                    grad_flat[j * 2 + 1] -= total_w * g[1]

    return grad_flat


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """Hard-disk relaxation: push overlapping circles apart."""
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        d = cdist(c, c)
        np.fill_diagonal(d, 1e-12)
        diff = c[:, None, :] - c[None, :, :]
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _active_set_refine(c, r, max_iter=30, tol=1e-5):
    """
    QCQP-style active-set refinement: identify kissing graph,
    then do focused optimization on active constraints.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        # Active pairs: distance close to 2*r_cur
        active_pairs = np.argwhere(d < 2 * r_cur + tol)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]

        # Active walls
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol:
                    active_walls.append((i, k))

        if len(active_pairs) == 0 and len(active_walls) == 0:
            break

        # Use very low temperature L-BFGS to focus on active constraints
        t_soft = max(1.0, 5 * tol / max(r_cur, 1e-6))
        x = c.ravel()
        res = minimize(
            _soft_min_radius_neg,
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t_soft),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            break

    return c, r_cur


def _init_hexagonal(rng):
    """Hexagonal packing initialization."""
    # Hexagonal lattice with spacing ~ 1/sqrt(50) * 2
    # Try to fit hexagonal arrangement
    c = []
    # Use a hexagonal grid
    dx = 0.145
    dy = dx * np.sqrt(3) / 2
    row = 0
    while len(c) < N:
        x_start = -0.5 + dx / 2 + (row % 2) * dx / 2
        for col in range(7):
            x = x_start + col * dx
            y = -0.5 + dy / 2 + row * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                c.append([x, y])
            if len(c) >= N:
                break
        row += 1
    c = np.array(c[:N])
    c += rng.uniform(-0.005, 0.005, c.shape)
    return np.clip(c, -0.49, 0.49)


def _init_7x7_plus():
    """7x7 grid (49 points) + 50th at centre offset."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.025, 0.025]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.015, 0.015, c.shape)
    return np.clip(c, -0.49, 0.49)


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """Gradually increase target radius R."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 12:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
        if (time.time() - t0) > DEADLINE - 15:
            break
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=15.0):
    """L-BFGS-B polish with decreasing temperature."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [80.0, 40.0, 20.0, 10.0, 5.0, 2.0]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx, tt=t: _soft_min_radius_neg(xx, tt),
            x,
            jac=lambda xx, tt=t: _soft_min_radius_grad(xx, tt),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()

    return best_x.reshape(N, 2), best_r


def _final_local_search(c, t0, time_budget=8.0):
    """Final local search with small perturbations and re-optimization."""
    c_best = c.copy()
    r_best = _radius_fast(c)
    rng = np.random.default_rng(12345)

    while (time.time() - t0) < t0 + time_budget:
        # Small random perturbation
        c_pert = c_best + rng.normal(0, 0.002, c_best.shape)
        c_pert = np.clip(c_pert, -0.499, 0.499)

        # Quick L-BFGS polish
        x = c_pert.ravel()
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, 5.0),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, 5.0),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'gtol': 1e-10}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > r_best:
            r_best = r_new
            c_best = c_new

    return c_best, r_best


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple initializations
    inits = [
        _init_hexagonal(rng),
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_random(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: Hard-disk radius expansion for each init
    for start in inits:
        if (time.time() - t0) > DEADLINE - 25:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0004, 600, 0.012, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 20:
        budget = min(18.0, DEADLINE - 20 - (time.time() - t0))
        c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=budget)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Active-set refinement
    if (time.time() - t0) < DEADLINE - 8:
        c_as, r_as = _active_set_refine(best_c, best_r, max_iter=40)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as

    # Phase 4: Final local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _final_local_search(best_c, t0, time_budget=min(7.0, DEADLINE - 3 - (time.time() - t0)))
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c

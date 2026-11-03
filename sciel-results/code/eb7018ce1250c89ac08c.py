"""Pack 50 equal circles in a unit square, maximize common radius.

Uses a multi-stage approach:
1. Multiple diverse initializations
2. Force-directed hard-disk relaxation with radius growing
3. Langevin dynamics to escape local minima
4. Active-set Newton refinement on the kissing graph
5. Multi-temperature L-BFGS-B soft-min polish
6. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """Differentiable soft-min surrogate for the radius."""
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft


def _soft_min_radius_neg(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)


def _soft_min_radius_grad(x, t):
    """Gradient of negative soft-min radius w.r.t. x (flattened centres)."""
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
    grad = np.zeros(2 * N)
    # Wall gradients
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad[:2 * N] += (exps[:2 * N] / Z) * w_grad
    # Pair gradients
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)
    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 0:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad[2*i:2*i+2] += total_w * g
                    grad[2*j:2*j+2] -= total_w * g
    return grad


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """Vectorized hard-disk relaxation with wall constraints."""
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _langevin_refine(c, r_cur, t_start, t_end, n_steps, dt, t0):
    """Langevin dynamics to escape local minima."""
    c = c.copy()
    best_c = c.copy()
    best_r = r_cur
    for step in range(n_steps):
        if (time.time() - t0) > DEADLINE - 5:
            break
        T = t_start + (t_end - t_start) * (step / max(n_steps - 1, 1))
        # Compute forces
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        f = np.zeros_like(c)
        # Pair forces: push apart if too close
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r_cur * 1.1:
                    dij = c[i] - c[j]
                    norm = d[i, j]
                    if norm > 1e-12:
                        f[i] += dij / norm
                        f[j] -= dij / norm
        # Wall forces
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0.5 - r_cur * 1.1:
                    f[i, k] -= 1.0
                elif c[i, k] < -0.5 + r_cur * 1.1:
                    f[i, k] += 1.0
        # Langevin step
        noise = np.random.randn(N, 2) * np.sqrt(2 * T * dt)
        c += dt * f + noise
        c = np.clip(c, -0.5, 0.5)
        r_new = _radius_fast(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    return best_c, best_r


def _active_set_newton(c, r, max_iter=100, tol=1e-10):
    """
    Active-set Newton refinement: identify the kissing graph, then solve
    for displacements that maintain active constraints while increasing r.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for outer in range(max_iter):
        if (time.time() - t0) > DEADLINE - 5:
            break
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        tol_d = max(1e-8, 1e-5 * r_cur)
        active_pairs = np.argwhere(d < 2 * r_cur + tol_d)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol_d:
                    active_walls.append((i, k))
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        # Build displacement
        disp = np.zeros_like(c)
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            deficit = max(2 * r_cur - d[i, j], 0.0)
            step = 0.5 * deficit / norm
            disp[i] += step * dij
            disp[j] -= step * dij
        for (i, k) in active_walls:
            deficit = max(r_cur - wall_dist[i, k], 0.0)
            sign = -np.sign(c[i, k])
            if c[i, k] == 0:
                sign = 1.0
            disp[i, k] += deficit * sign
        max_disp = np.max(np.linalg.norm(disp, axis=1))
        if max_disp < 1e-12:
            break
        disp = disp / max_disp * min(0.3 * r_cur, 1e-5)
        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            found = False
            for scale in [0.5, 0.25, 0.1, 0.05, 0.01]:
                c_try = c + scale * disp
                c_try = np.clip(c_try, -0.5, 0.5)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-12:
                    c = c_try
                    r_cur = r_try
                    found = True
                    break
            if not found:
                break
    return c, r_cur


def _final_local_search(c, r, max_iter=200, step_scale=1e-6):
    """Deterministic local search with adaptive step size."""
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)
    step = step_scale
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in directions:
                c_try = c.copy()
                c_try[i] += step * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    return c, r_cur


def _init_grid_7x7():
    """7x7 grid (49 points) + 50th near centre."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
    return c


def _init_hexagonal(rng):
    """Hexagonal packing initialization."""
    c = np.zeros((N, 2))
    dx = 0.12
    dy = dx * np.sqrt(3) / 2
    row = 0
    col = 0
    idx = 0
    while idx < N:
        x = (col + 0.5 * (row % 2)) * dx - 0.5
        y = row * dy - 0.5
        if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
            c[idx] = [x, y]
            idx += 1
        col += 1
        if col > 8:
            col = 0
            row += 1
    c += rng.uniform(-0.005, 0.005, c.shape)
    return c


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _init_clustered(rng):
    """Initialize with clusters to encourage diverse packings."""
    c = np.zeros((N, 2))
    n_clusters = 5
    centers = rng.uniform(-0.3, 0.3, (n_clusters, 2))
    for i in range(N):
        k = i % n_clusters
        c[i] = centers[k] + rng.normal(0, 0.08, 2)
    c = np.clip(c, -0.45, 0.45)
    return c


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """Gradually increase target radius with hard-disk relaxation."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 15:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=15.0):
    """L-BFGS-B polish with decreasing temperature."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    temps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-16}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()
    return best_x.reshape(N, 2), best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
        _init_clustered(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: Hard-disk radius expansion for each init
    for start in inits:
        if (time.time() - t0) > DEADLINE - 40:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: Langevin refinement to escape local minima
    if (time.time() - t0) < DEADLINE - 25:
        c_lang, r_lang = _langevin_refine(best_c, best_r, 0.001, 0.0001, 2000, 0.001, t0)
        if r_lang > best_r:
            best_r = r_lang
            best_c = c_lang

    # Phase 3: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 20:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(15.0, DEADLINE - 20 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 4: Active-set Newton refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_newton, r_newton = _active_set_newton(best_c, best_r, max_iter=100)
        if r_newton > best_r:
            best_r = r_newton
            best_c = c_newton

    # Phase 5: Another L-BFGS pass with lower temperatures
    if (time.time() - t0) < DEADLINE - 8:
        c_polished2, r_polished2 = _lbfgs_polish(
            best_c, t0,
            time_budget=min(8.0, DEADLINE - 8 - (time.time() - t0))
        )
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=200, step_scale=1e-6)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_c

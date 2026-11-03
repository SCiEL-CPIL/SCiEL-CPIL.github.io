"""Pack 50 equal circles in a unit square, maximize common radius.

Uses a multi-stage approach:
1. Multiple initializations (grid, jittered grid, hexagonal, random)
2. Hard-disk radius expansion to get close to optimal
3. SLSQP epigraph polish: directly maximize t subject to
   pairwise distance >= 2t and wall distance >= t constraints.
4. L-BFGS-B soft-min polish with adaptive temperature
5. High-precision final local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 118.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances. As t -> 0 this approaches the true min radius.
    """
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

    grad_flat = np.zeros(2 * N)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
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
                    grad_flat[i * 2] += total_w * g[0]
                    grad_flat[i * 2 + 1] += total_w * g[1]
                    grad_flat[j * 2] -= total_w * g[0]
                    grad_flat[j * 2 + 1] -= total_w * g[1]

    return grad_flat


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square. Vectorized.
    """
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


def _slsqp_epigraph_polish(c, r, time_budget=25.0):
    """
    SLSQP epigraph formulation: maximize t subject to
    ||c_i - c_j|| >= 2t for all i<j,
    0.5 - |c_ik| >= t for all i,k.
    
    Variables: x = [c_1, c_2, ..., c_N, t] (2N+1 variables)
    Objective: -t (minimize)
    Constraints (inequality, >= 0):
      ||c_i - c_j||^2 - 4t^2 >= 0  (pairwise)
      (0.5 - |c_ik|)^2 - t^2 >= 0  (wall)
    
    This directly optimizes the exact objective with SLSQP's constraint handling.
    """
    def obj(x):
        return -x[-1]  # minimize -t

    def obj_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        c = x[:2 * N].reshape(N, 2)
        t = x[-1]
        cons = []
        # Pairwise constraints: ||c_i - c_j||^2 - 4t^2 >= 0
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2) - 4*x[-1]**2,
                    'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
                })
        # Wall constraints: (0.5 - |c_ik|)^2 - t^2 >= 0
        for i in range(N):
            for k in range(2):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, k=k: (0.5 - abs(x[2*i+k]))**2 - x[-1]**2,
                    'jac': lambda x, i=i, k=k: _wall_jac(x, i, k)
                })
        return cons

    def _pair_jac(x, i, j):
        g = np.zeros_like(x)
        di = x[2*i:2*i+2] - x[2*j:2*j+2]
        g[2*i:2*i+2] = 2 * di
        g[2*j:2*j+2] = -2 * di
        g[-1] = -8 * x[-1]
        return g

    def _wall_jac(x, i, k):
        g = np.zeros_like(x)
        val = x[2*i+k]
        sign = np.sign(val) if val != 0 else 1.0
        g[2*i+k] = -2 * sign * (0.5 - abs(val))
        g[-1] = -2 * x[-1]
        return g

    x0 = np.concatenate([c.ravel(), [r]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.01, 0.1)]
    
    best_x = x0.copy()
    best_r = r
    t_start = time.time()
    
    # Run SLSQP with increasing iterations and tighter tolerances
    for maxiter, ftol in [(200, 1e-10), (500, 1e-12), (1000, 1e-14)]:
        if time.time() - t_start > time_budget:
            break
        try:
            res = minimize(
                obj, x0, jac=obj_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success or res.fun < -best_r:
                r_new = -res.fun
                c_new = res.x[:2*N].reshape(N, 2)
                # Validate
                r_actual = _radius_fast(c_new)
                if r_actual > best_r - 1e-12:
                    best_r = max(r_actual, r_new if np.isfinite(r_new) else r_actual)
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    
    return best_x[:2*N].reshape(N, 2), best_r


def _voronoi_style_refine(c, r, max_iter=30, tol=1e-6):
    """
    Voronoi-style active-set refinement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        tol_d = max(1e-5, 2e-4 * r_cur)
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
            disp[i, k] += 0.5 * deficit * sign

        max_disp = np.max(np.linalg.norm(disp, axis=1))
        if max_disp > 1e-12:
            disp = disp / max_disp * min(0.5 * r_cur, 1e-4)

        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            c_try = c + 0.1 * disp
            c_try = np.clip(c_try, -0.5, 0.5)
            r_try = _radius_fast(c_try)
            if r_try > r_cur + 1e-12:
                c = c_try
                r_cur = r_try
            else:
                break

    return c, r_cur


def _final_local_search(c, r, max_iter=200, step_scale=1e-5):
    """
    Deterministic local search with adaptive step size.
    """
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
            if step < 1e-9:
                break
    return c, r_cur


def _init_7x7_plus():
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


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


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


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """
    Gradually increase the target radius R from R0 to R1,
    running hard-disk relaxation at each step.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 10:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=10.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                lambda xx: _soft_min_radius_neg(xx, t),
                x,
                jac=lambda xx: _soft_min_radius_grad(xx, t),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_x = res.x.copy()
                x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple initializations
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]

    best_c = None
    best_r = -1.0

    for start in inits:
        if (time.time() - t0) > DEADLINE - 40:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 800, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 35:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(15.0, DEADLINE - 35 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: SLSQP epigraph polish - the key improvement
    if (time.time() - t0) < DEADLINE - 20:
        budget = min(20.0, DEADLINE - 20 - (time.time() - t0))
        if budget > 5.0:
            c_slsqp, r_slsqp = _slsqp_epigraph_polish(best_c, best_r, time_budget=budget)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp

    # Phase 4: Voronoi-style active-set refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=30)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 5: Another SLSQP pass with tighter tolerance
    if (time.time() - t0) < DEADLINE - 8:
        budget = min(10.0, DEADLINE - 8 - (time.time() - t0))
        if budget > 3.0:
            c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(best_c, best_r, time_budget=budget)
            if r_slsqp2 > best_r:
                best_r = r_slsqp2
                best_c = c_slsqp2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=200, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c

"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy:
1. Multiple diverse initializations
2. Hard-disk relaxation with adaptive radius growth
3. Active-set refinement: identify critical constraints, solve local QCQP
4. L-BFGS-B soft-min polish
5. SLSQP epigraph polish
6. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 143.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """Differentiable soft-min of all constraints."""
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
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)
    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 1e-15:
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
    """Hard-disk relaxation with vectorized forces."""
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
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _active_set_refine(c, r, t0, time_budget=25.0):
    """
    Active-set refinement: identify the most critical constraints and
    optimize only the subset of circles involved in them.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    best_c = c.copy()
    best_r = r_cur

    elapsed = time.time() - t0
    if elapsed > t0 + time_budget:
        return best_c, best_r

    # Find active constraints (those within 2% of the minimum)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0
    wall_dists = 0.5 - np.abs(c)
    all_constr = np.concatenate([wall_dists.ravel(), pair_dists.ravel()])
    threshold = r_cur * 1.02

    # Identify which circles are in active constraints
    active_circles = set()
    for i in range(N):
        for k in range(2):
            if wall_dists[i, k] <= threshold:
                active_circles.add(i)
    for i in range(N):
        for j in range(i + 1, N):
            if pair_dists[i, j] <= threshold:
                active_circles.add(i)
                active_circles.add(j)

    if not active_circles:
        return best_c, best_r

    active_idx = sorted(active_circles)
    n_active = len(active_idx)

    # Optimize only the active circles
    c_sub = c[active_idx].copy()
    c_fixed = c[~np.isin(np.arange(N), active_idx)].copy()

    def obj_active(x):
        c_try = c.copy()
        c_try[active_idx] = x.reshape(n_active, 2)
        return -_radius_fast(c_try)

    def grad_active(x):
        c_try = c.copy()
        c_try[active_idx] = x.reshape(n_active, 2)
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for k in range(len(x)):
            xp = x.copy()
            xp[k] += eps
            xm = x.copy()
            xm[k] -= eps
            g[k] = (obj_active(xp) - obj_active(xm)) / (2 * eps)
        return g

    x0 = c_sub.ravel()
    bounds = [(-0.5, 0.5)] * (2 * n_active)

    try:
        res = minimize(
            obj_active,
            x0,
            jac=grad_active,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        if res.success:
            c_new = c.copy()
            c_new[active_idx] = res.x.reshape(n_active, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
    except Exception:
        pass

    return best_c, best_r


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """
    SLSQP polish using epigraph formulation.
    """
    x0 = np.concatenate([c_init.ravel(), [r_init]])

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        cons = []
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                cons.append(d[i, j] - 2 * r)
        wall_dist = 0.5 - np.abs(c)
        for i in range(N):
            for k in range(2):
                cons.append(wall_dist[i, k] - r)
        return cons

    def constraints_jac(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        n_cons = N * (N - 1) // 2 + 2 * N
        J = np.zeros((n_cons, 2 * N + 1))
        d = cdist(c, c)
        np.fill_diagonal(d, 1.0)
        row = 0
        for i in range(N):
            for j in range(i + 1, N):
                dij = c[i] - c[j]
                norm = d[i, j]
                if norm > 1e-12:
                    J[row, 2 * i] = dij[0] / norm
                    J[row, 2 * i + 1] = dij[1] / norm
                    J[row, 2 * j] = -dij[0] / norm
                    J[row, 2 * j + 1] = -dij[1] / norm
                J[row, -1] = -2.0
                row += 1
        for i in range(N):
            for k in range(2):
                sign = np.sign(c[i, k])
                if sign == 0:
                    sign = 1.0
                J[row, 2 * i + k] = -sign
                J[row, -1] = -1.0
                row += 1
        return J

    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]

    def fun_con(x):
        return np.array(constraints(x))

    def jac_con(x):
        return constraints_jac(x)

    cons = [{'type': 'ineq', 'fun': fun_con, 'jac': jac_con}]

    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.success:
            c_new = res.x[:2 * N].reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > _radius_fast(c_init):
                return c_new, r_new
    except Exception:
        pass

    return c_init, _radius_fast(c_init)


def _final_local_search(c, r, max_iter=300, step_scale=1e-5):
    """Deterministic local search with adaptive step size."""
    c = c.copy()
    r_cur = _radius_fast(c)

    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    fine_dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    for it in range(max_iter):
        improved = False
        for i in range(N):
            for d in fine_dirs:
                for scale in [step_scale, step_scale * 0.5, step_scale * 2]:
                    c_try = c.copy()
                    c_try[i] += scale * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-12:
                        c = c_try
                        r_cur = r_try
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try smaller step
            step_scale *= 0.7
            if step_scale < 1e-8:
                break
    return c, r_cur


def _init_7x7_plus():
    """7x7 grid (49 points) + 50th near centre."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.015, 0.015, c.shape)
    return c


def _init_hexagonal(rng):
    """Hexagonal close-packing initialization."""
    s = 0.12
    pts = []
    for row in range(8):
        y = -0.45 + row * s * np.sqrt(3) / 2
        offset = (s / 2) if (row % 2 == 1) else 0.0
        for col in range(8):
            x = -0.45 + col * s + offset
            if abs(x) <= 0.45 and abs(y) <= 0.45:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > N:
        idx = rng.permutation(len(pts))[:N]
        pts = pts[idx]
    elif len(pts) < N:
        extra = rng.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts += rng.uniform(-0.01, 0.01, pts.shape)
    return np.clip(pts, -0.45, 0.45)


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """Gradually increase the target radius R."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 20:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=20.0):
    """L-BFGS-B polish using soft-min radius with adaptive temperature."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.8]
    maxiter_per_temp = 300

    for t in temps:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break
        mi = maxiter_per_temp if t < 5.0 else maxiter_per_temp // 2
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': mi, 'gtol': 1e-12, 'ftol': 1e-16}
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
        if (time.time() - t0) > DEADLINE - 50:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 500, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 40:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(25.0, DEADLINE - 40 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Active-set refinement
    if (time.time() - t0) < DEADLINE - 25:
        c_active, r_active = _active_set_refine(
            best_c, best_r, t0,
            time_budget=min(20.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_active > best_r:
            best_r = r_active
            best_c = c_active

    # Phase 4: SLSQP Epigraph Polish
    if (time.time() - t0) < DEADLINE - 15:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(25.0, DEADLINE - 15 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 5: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 5:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=200, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 1:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=150, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c

"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. Progressive beta L-BFGS-B soft-min polish
4. Full-constraint SLSQP epigraph polish with active-set reduction
5. Active-set refinement on the kissing graph
6. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius_neg(x, t):
    """
    Differentiable surrogate: negative soft-min of all constraints.
    As t -> 0 this approaches the true negative min radius.
    """
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
    soft = m - t * np.log(Z)
    return -soft


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

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    for idx in range(2 * N):
        i, k = divmod(idx, 2)
        s = sign_c[i, k]
        if s == 0:
            s = 1.0
        grad[idx] += -(exps[idx] / Z) * s

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)

    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = 2 * N + i * N + j
            idx_ji = 2 * N + j * N + i
            w_ij = exps[idx_ij] / Z
            w_ji = exps[idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 1e-15:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad[2*i] += total_w * g[0]
                    grad[2*i+1] += total_w * g[1]
                    grad[2*j] -= total_w * g[0]
                    grad[2*j+1] -= total_w * g[1]

    return grad


def _slsqp_full_polish(c, r_init, t0, time_budget=35.0):
    """
    SLSQP with full constraint set: all pairs and all walls.
    Direct max-min radius optimization via epigraph formulation.
    """
    c = c.copy()
    r = _radius_fast(c)
    best_c = c.copy()
    best_r = r

    # Build all pair constraints
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_walls = [(i, k) for i in range(N) for k in range(2)]

    constraints = []

    for (i, j) in all_pairs:
        def cfun(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            rr = x[-1]
            return np.sqrt(np.sum((ci - cj)**2)) - 2*rr

        def cjac(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            g = np.zeros(2 * N + 1)
            dij = ci - cj
            norm = np.sqrt(np.sum(dij**2))
            if norm > 1e-12:
                g[2*i] = dij[0] / norm
                g[2*i+1] = dij[1] / norm
                g[2*j] = -dij[0] / norm
                g[2*j+1] = -dij[1] / norm
            g[-1] = -2.0
            return g

        constraints.append({'type': 'ineq', 'fun': cfun, 'jac': cjac})

    for (i, k) in all_walls:
        def cfun(x, i=i, k=k):
            c_ik = x[2*i + k]
            rr = x[-1]
            return 0.5 - np.abs(c_ik) - rr

        def cjac(x, i=i, k=k):
            c_ik = x[2*i + k]
            g = np.zeros(2 * N + 1)
            s = np.sign(c_ik)
            if s == 0:
                s = 1.0
            g[2*i + k] = -s
            g[-1] = -1.0
            return g

        constraints.append({'type': 'ineq', 'fun': cfun, 'jac': cjac})

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g

    x0 = np.concatenate([best_c.ravel(), [best_r]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

    # Multiple passes with increasing maxiter
    for maxiter in [300, 800, 1500]:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-15}
            )
            if res.success or res.fun < -best_r:
                r_new = res.x[-1]
                c_new = res.x[:2*N].reshape(N, 2)
                r_verified = _radius_fast(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_c = c_new.copy()
                    x0 = res.x.copy()
        except Exception:
            pass

    return best_c, best_r


def _slsqp_active_set_polish(c, r_init, t0, time_budget=20.0):
    """
    SLSQP with active-set reduction: only constrain pairs and walls
    that are close to being active.  Used as a quick refinement pass.
    """
    c = c.copy()
    r = _radius_fast(c)

    def _get_active_constraints(c, r, tol=0.002):
        """Return (active_pairs, active_walls) within tol of the current radius."""
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall = 0.5 - np.abs(c)

        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + tol:
                    active_pairs.append((i, j))

        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall[i, k] < r + tol:
                    active_walls.append((i, k))

        return active_pairs, active_walls

    best_c = c.copy()
    best_r = r

    for pass_num in range(2):
        if (time.time() - t0) > t0 + time_budget:
            break

        active_pairs, active_walls = _get_active_constraints(best_c, best_r, tol=0.001)

        # Build constraints
        constraints = []

        for (i, j) in active_pairs:
            def cfun(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                rr = x[-1]
                return np.sqrt(np.sum((ci - cj)**2)) - 2*rr

            def cjac(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                rr = x[-1]
                g = np.zeros(2 * N + 1)
                dij = ci - cj
                norm = np.sqrt(np.sum(dij**2))
                if norm > 1e-12:
                    g[2*i] = dij[0] / norm
                    g[2*i+1] = dij[1] / norm
                    g[2*j] = -dij[0] / norm
                    g[2*j+1] = -dij[1] / norm
                g[-1] = -2.0
                return g

            constraints.append({'type': 'ineq', 'fun': cfun, 'jac': cjac})

        for (i, k) in active_walls:
            def cfun(x, i=i, k=k):
                c_ik = x[2*i + k]
                rr = x[-1]
                return 0.5 - np.abs(c_ik) - rr

            def cjac(x, i=i, k=k):
                c_ik = x[2*i + k]
                rr = x[-1]
                g = np.zeros(2 * N + 1)
                s = np.sign(c_ik)
                if s == 0:
                    s = 1.0
                g[2*i + k] = -s
                g[-1] = -1.0
                return g

            constraints.append({'type': 'ineq', 'fun': cfun, 'jac': cjac})

        if not constraints:
            break

        def objective(x):
            return -x[-1]

        def objective_grad(x):
            g = np.zeros(2 * N + 1)
            g[-1] = -1.0
            return g

        x0 = np.concatenate([best_c.ravel(), [best_r]])
        bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

        for maxiter in [200, 500]:
            if (time.time() - t0) > t0 + time_budget:
                break
            try:
                res = minimize(
                    objective,
                    x0,
                    jac=objective_grad,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': 1e-14}
                )
                if res.fun < -best_r:
                    r_new = res.x[-1]
                    c_new = res.x[:2*N].reshape(N, 2)
                    r_verified = _radius_fast(c_new)
                    if r_verified > best_r:
                        best_r = r_verified
                        best_c = c_new.copy()
                        x0 = res.x.copy()
            except Exception:
                pass

    return best_c, best_r


def _lbfgs_polish(c_init, r_init, t0, time_budget=25.0):
    """
    L-BFGS-B polish using soft-min surrogate with progressive beta.
    Higher betas give tighter approximation to the true min.
    """
    best_c = c_init.copy()
    best_r = r_init

    # Progressive beta: start moderate, go very high for tight approximation
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0]

    for beta in betas:
        if (time.time() - t0) > t0 + time_budget:
            break

        def obj(x):
            return _soft_min_radius_neg(x, beta)

        def obj_grad(x):
            return _soft_min_radius_grad(x, beta)

        try:
            res = minimize(
                obj,
                best_c.ravel(),
                jac=obj_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            continue

    return best_c, best_r


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square.  Vectorized.
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


def _kissing_graph(c, r, tol=1e-3):
    """Find active constraints."""
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    wall = 0.5 - np.abs(c)

    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(d[i, j] - 2 * r) < tol:
                active_pairs.append((i, j))

    active_walls = []
    for i in range(N):
        for k in range(2):
            if wall[i, k] < r + tol:
                active_walls.append((i, k))

    return active_pairs, active_walls


def _active_set_refine(c, r, max_iter=100, tol=1e-10):
    """
    Active-set refinement: identify the kissing graph, then solve
    a local problem to maximize the minimum constraint.
    """
    c = c.copy()
    r_cur = _radius_fast(c)

    for _ in range(max_iter):
        active_pairs, active_walls = _kissing_graph(c, r_cur, tol=1e-4)

        if not active_pairs and not active_walls:
            break

        grad = np.zeros((N, 2))

        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm > 1e-12:
                g = dij / norm
                grad[i] += 0.5 * g
                grad[j] -= 0.5 * g

        for (i, k) in active_walls:
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            grad[i, k] += -0.5 * sign

        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12:
            break
        grad /= norm_grad

        best_step = 0
        best_r = r_cur
        for step in np.linspace(0, 0.002, 20):
            c_try = c + step * grad
            c_try = np.clip(c_try, -0.5, 0.5)
            r_try = _radius_fast(c_try)
            if r_try > best_r:
                best_r = r_try
                best_step = step

        if best_step == 0:
            break

        c = c + best_step * grad
        c = np.clip(c, -0.5, 0.5)
        r_cur = _radius_fast(c)

        if best_step < tol:
            break

    return c, _radius_fast(c)


def _final_local_search(c, r, max_iter=500, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions (including radial and diagonal), keeping the
    best improvement.  Uses adaptive step scaling.
    """
    c = c.copy()
    r_cur = _radius_fast(c)

    # 32 directions: 24 angular + 4 cardinal + 4 diagonal
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add cardinal and diagonal directions explicitly for better coverage
    extra_dirs = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ])
    extra_dirs = extra_dirs / np.linalg.norm(extra_dirs, axis=1, keepdims=True)
    dirs = np.vstack([dirs, extra_dirs])

    current_step = step_scale

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try multiple step sizes for adaptive refinement
            for scale_factor in [1.0, 0.5, 0.2]:
                step = current_step * scale_factor
                for d in dirs:
                    c_try = c.copy()
                    c_try[i] += step * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-14:
                        c = c_try
                        r_cur = r_try
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Reduce step size and try again
            current_step *= 0.5
            if current_step < 1e-8:
                break
            continue
        # On improvement, keep step size or slightly reduce
        current_step *= 0.8

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
    """
    Gradually increase the target radius R from R0 to R1,
    running hard-disk relaxation at each step.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 30:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple initializations with varied structures
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]

    # Add a second hexagonal init with different spacing
    s2 = 0.11
    pts2 = []
    for row in range(9):
        y = -0.46 + row * s2 * np.sqrt(3) / 2
        offset = (s2 / 2) if (row % 2 == 1) else 0.0
        for col in range(9):
            x = -0.46 + col * s2 + offset
            if abs(x) <= 0.46 and abs(y) <= 0.46:
                pts2.append([x, y])
    pts2 = np.array(pts2)
    if len(pts2) > N:
        idx2 = rng.permutation(len(pts2))[:N]
        pts2 = pts2[idx2]
    elif len(pts2) < N:
        extra2 = rng.uniform(-0.4, 0.4, (N - len(pts2), 2))
        pts2 = np.vstack([pts2, extra2])
    pts2 += rng.uniform(-0.01, 0.01, pts2.shape)
    inits.append(np.clip(pts2, -0.45, 0.45))

    best_c = None
    best_r = -1.0

    # Phase 1: Quick hard-disk expansion on all initializations
    for start in inits:
        if (time.time() - t0) > DEADLINE - 45:
            break
        c, r = _grow_radius(start, 0.04, 0.073, 0.0004, 400, 0.02, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish (progressive beta, higher values)
    if (time.time() - t0) < DEADLINE - 45:
        c_lbfgs, r_lbfgs = _lbfgs_polish(
            best_c, best_r, t0,
            time_budget=min(25.0, DEADLINE - 45 - (time.time() - t0))
        )
        if r_lbfgs > best_r:
            best_r = r_lbfgs
            best_c = c_lbfgs

    # Phase 3: Full-constraint SLSQP epigraph polish (strongest refinement)
    if (time.time() - t0) < DEADLINE - 25:
        c_slq, r_slq = _slsqp_full_polish(
            best_c, best_r, t0,
            time_budget=min(35.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_slq > best_r:
            best_r = r_slq
            best_c = c_slq

    # Phase 4: Active-set SLSQP for quick additional refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_aslq, r_aslq = _slsqp_active_set_polish(
            best_c, best_r, t0,
            time_budget=min(15.0, DEADLINE - 15 - (time.time() - t0))
        )
        if r_aslq > best_r:
            best_r = r_aslq
            best_c = c_aslq

    # Phase 5: Active-set refinement
    if (time.time() - t0) < DEADLINE - 8:
        c_as, r_as = _active_set_refine(best_c, best_r, max_iter=150)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as

    # Phase 6: Final high-precision local search (two passes)
    if (time.time() - t0) < DEADLINE - 4:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=500, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        if (time.time() - t0) < DEADLINE - 2:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=400, step_scale=5e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

        if (time.time() - t0) < DEADLINE - 1:
            c_ls3, r_ls3 = _final_local_search(best_c, best_r, max_iter=300, step_scale=2e-6)
            if r_ls3 > best_r:
                best_r = r_ls3
                best_c = c_ls3

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c

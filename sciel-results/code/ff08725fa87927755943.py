"""Pack 50 equal circles in a unit square, maximize common radius.

Combined approach:
1. Multiple diverse initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. Multi-temperature L-BFGS-B soft-min polish with adaptive temperature schedule
4. SLSQP Epigraph Polish: Maximizes r subject to d_ij >= 2r and w_ik >= r
5. Active-set refinement on the kissing graph
6. Final high-precision deterministic local search
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
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances.  As t -> 0 this approaches the true min radius.
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


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """
    SLSQP polish using epigraph formulation.
    Variables: [x_1, y_1, ..., x_N, y_N, r]
    Objective: minimize -r
    Constraints:
      1. d_ij >= 2r  for all i < j
      2. w_ik >= r   for all i, k
    """
    x0 = np.concatenate([c_init.ravel(), [r_init]])
    
    # Precompute indices for constraints
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j))
            
    wall_indices = []
    for i in range(N):
        for k in range(2):
            wall_indices.append((i, k))

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        cons = []
        
        # Pair constraints: d_ij - 2r >= 0
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        for (i, j) in pair_indices:
            cons.append(d[i, j] - 2 * r)
            
        # Wall constraints: w_ik - r >= 0
        wall_dist = 0.5 - np.abs(c)
        for (i, k) in wall_indices:
            cons.append(wall_dist[i, k] - r)
            
        return cons

    def constraints_jac(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        # Jacobian shape: (num_constraints, num_vars)
        # num_vars = 2N + 1
        J = np.zeros((len(pair_indices) + len(wall_indices), 2*N + 1))
        
        # Pair constraints
        d = cdist(c, c)
        np.fill_diagonal(d, 1.0) # avoid div by zero
        row = 0
        for (i, j) in pair_indices:
            dij = c[i] - c[j]
            norm = d[i, j]
            if norm > 1e-12:
                # d/dc_i = (c_i - c_j) / d
                J[row, 2*i] = dij[0] / norm
                J[row, 2*i+1] = dij[1] / norm
                # d/dc_j = (c_j - c_i) / d
                J[row, 2*j] = -dij[0] / norm
                J[row, 2*j+1] = -dij[1] / norm
            # d/dr = -2
            J[row, -1] = -2.0
            row += 1
            
        # Wall constraints
        for (i, k) in wall_indices:
            # d/dc_ik = -sign(c_ik)
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            J[row, 2*i + k] = -sign
            # d/dr = -1
            J[row, -1] = -1.0
            row += 1
            
        return J

    # Bounds: centers in [-0.5, 0.5], r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2*N) + [(0, 0.5)]

    # Constraints for SLSQP
    def fun_con(x):
        return np.array(constraints(x))
        
    def jac_con(x):
        return constraints_jac(x)
        
    cons = [{'type': 'ineq', 'fun': fun_con, 'jac': jac_con}]

    c_cur = c_init.copy()
    r_cur = r_init

    # First SLSQP pass: moderate tolerance
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 800, 'ftol': 1e-12}
        )
        
        if res.success:
            c_new = res.x[:2*N].reshape(N, 2)
            r_new = res.x[-1]
            r_check = _radius_fast(c_new)
            if r_check > r_cur:
                r_cur = r_check
                c_cur = c_new
    except Exception:
        pass

    # Second SLSQP pass: tighter tolerance, starting from improved solution
    elapsed = time.time() - t0
    if elapsed < t0 + time_budget - 5:
        x2 = np.concatenate([c_cur.ravel(), [r_cur]])
        try:
            res2 = minimize(
                objective,
                x2,
                jac=objective_grad,
                bounds=bounds,
                constraints=cons,
                method='SLSQP',
                options={'maxiter': 600, 'ftol': 1e-15}
            )
            
            if res2.success:
                c_new2 = res2.x[:2*N].reshape(N, 2)
                r_new2 = res2.x[-1]
                r_check2 = _radius_fast(c_new2)
                if r_check2 > r_cur:
                    r_cur = r_check2
                    c_cur = c_new2
        except Exception:
            pass

    return c_cur, r_cur


def _voronoi_style_refine(c, r, max_iter=50, step_scale=1e-4):
    """
    Voronoi-style refinement: for each circle, compute the direction
    toward the farthest point in its Voronoi cell (approximated by
    the midpoint to the nearest neighbor), and nudge it that way.
    This pushes circles into the center of their available space.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Find nearest neighbors
            dists = np.sqrt(np.sum((c - c[i])**2, axis=1))
            dists[i] = np.inf
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            
            # Direction from nearest neighbor to circle i (push away)
            if nearest_dist > 1e-12:
                away_dir = (c[i] - c[nearest_idx]) / nearest_dist
            else:
                away_dir = np.array([0.0, 0.0])
            
            # Also consider wall proximity
            wall_push = np.array([0.0, 0.0])
            for k in range(2):
                if c[i, k] > 0.0:
                    wall_push[k] += 1.0  # push left
                else:
                    wall_push[k] -= 1.0  # push right
            wall_push /= np.linalg.norm(wall_push) + 1e-15
            
            # Combined direction: away from nearest neighbor + slight wall avoidance
            combined = away_dir + 0.3 * wall_push
            norm_c = np.linalg.norm(combined)
            if norm_c > 1e-12:
                combined /= norm_c
            
            # Try nudging
            c_try = c.copy()
            c_try[i] += step_scale * combined
            c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
            r_try = _radius_fast(c_try)
            if r_try > r_cur + 1e-13:
                c = c_try
                r_cur = r_try
                improved = True
        
        if not improved:
            break
    
    return c, r_cur


def _final_local_search(c, r, max_iter=300, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in 8 directions, keeping the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)

    # Also add finer angular directions
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    fine_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    all_dirs = np.vstack([directions, fine_dirs])

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in all_dirs:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-12:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
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
    # Hexagonal lattice: spacing s, rows offset by s/2
    s = 0.12  # initial spacing
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
    while R <= R1 and (time.time() - t0) < DEADLINE - 15:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=20.0):
    """
    L-BFGS-B polish using soft-min radius with adaptive decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    # Adaptive temperature schedule: start high, decrease geometrically
    temps = [200.0, 100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.8]
    maxiter_per_temp = 400

    for t in temps:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break

        # For smaller temps, use more iterations
        if t < 5.0:
            mi = maxiter_per_temp
        else:
            mi = maxiter_per_temp // 2

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
        if (time.time() - t0) > DEADLINE - 40:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish (reduced budget to give more to SLSQP)
    if (time.time() - t0) < DEADLINE - 40:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(20.0, DEADLINE - 40 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: SLSQP Epigraph Polish (primary optimizer, generous budget)
    if (time.time() - t0) < DEADLINE - 20:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(45.0, DEADLINE - 20 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 4: Second SLSQP pass with even tighter tolerance
    if (time.time() - t0) < DEADLINE - 8:
        c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(12.0, DEADLINE - 8 - (time.time() - t0))
        )
        if r_slsqp2 > best_r:
            best_r = r_slsqp2
            best_c = c_slsqp2

    # Phase 5: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 6:
        c_vor, r_vor = _voronoi_style_refine(best_c, best_r, max_iter=40, step_scale=1e-4)
        if r_vor > best_r:
            best_r = r_vor
            best_c = c_vor

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
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

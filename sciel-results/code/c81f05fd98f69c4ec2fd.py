"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. SLSQP epigraph polish with active-set reduction
4. Active-set refinement on the kissing graph
5. Final high-precision deterministic local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _kissing_graph(c, r, tol=1e-3):
    """
    Find active constraints: pairs at distance ~2r and circles touching walls.
    Returns list of (i, j) for pairs and (i, k) for walls.
    """
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


def _soft_min_radius(c, temperature=0.01):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth surrogate for min(pair_dist/2, wall_dist).
    """
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    wall = 0.5 - np.abs(c)
    
    # All pairwise distances / 2
    pair_vals = d / 2.0
    # Wall distances
    wall_vals = wall.flatten()
    
    # Combine all values
    all_vals = np.concatenate([pair_vals.flatten(), wall_vals])
    
    # Soft min via log-sum-exp: -T * log(sum(exp(-x/T)))
    # Numerically stable version
    min_val = all_vals.min()
    shifted = all_vals - min_val
    # Use max for numerical stability
    max_shifted = np.max(shifted / temperature)
    log_sum = max_shifted + np.log(np.sum(np.exp(-(shifted / temperature) - max_shifted)))
    soft_min = min_val - temperature * log_sum
    
    return soft_min


def _soft_min_radius_grad(c, temperature=0.01):
    """
    Gradient of the soft-min radius with respect to circle centers.
    Uses the formula: grad soft_min = -sum_i w_i * grad(x_i) / sum_i w_i
    where w_i = exp(-x_i / T) and x_i are the constraint values.
    """
    c = c.reshape(-1, 2)
    n = len(c)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    wall = 0.5 - np.abs(c)
    
    # Collect all constraint values and their gradients
    # For pair (i,j): value = ||c_i - c_j|| / 2
    # d/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/d(c_j) = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # For wall (i,k): value = 0.5 - |c_ik|
    # d/d(c_ik) = -sign(c_ik)
    
    grad = np.zeros((n, 2))
    total_w = 0.0
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            val = norm / 2.0
            w = np.exp(-val / temperature)
            total_w += w
            g = dij / norm
            grad[i] += w * g / 2.0
            grad[j] -= w * g / 2.0
    
    # Wall
    for i in range(n):
        for k in range(2):
            val = wall[i, k]
            w = np.exp(-val / temperature)
            total_w += w
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            grad[i, k] += w * (-sign)
    
    if total_w > 0:
        grad /= total_w
    else:
        grad = np.zeros((n, 2))
    
    return grad


def _lbfgs_soft_min_polish(c_init, r_init, t0, time_budget=30.0):
    """
    L-BFGS-B polish using soft-min smooth surrogate.
    Maximize soft_min_radius(c) subject to c in [-0.5, 0.5]^2.
    """
    from scipy.optimize import minimize as sp_minimize
    
    best_c = c_init.copy()
    best_r = r_init
    best_soft = _soft_min_radius(best_c, 0.01)
    
    # Try multiple temperatures, decreasing
    for temp in [0.05, 0.02, 0.01, 0.005, 0.002]:
        if time.time() - t0 > t0 + time_budget:
            break
        
        def obj(x):
            c = x.reshape(N, 2)
            return -_soft_min_radius(c, temp)
        
        def obj_grad(x):
            c = x.reshape(N, 2)
            return -_soft_min_radius_grad(c, temp).flatten()
        
        x0 = best_c.ravel()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = sp_minimize(
                obj, x0, jac=obj_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            soft_new = _soft_min_radius(c_new, temp)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                best_soft = soft_new
        except Exception:
            pass
    
    return best_c, best_r


def _voronoi_style_refine(c, r, max_iter=200, tol=1e-12):
    """
    Voronoi-style refinement: for each circle, find its local clearance
    (minimum distance to neighbors and walls) and move it to increase
    that clearance. Iterate until convergence.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Find local clearance for circle i
            d = np.linalg.norm(c - c[i], axis=1)
            d[i] = np.inf
            wall_dist = np.min(0.5 - np.abs(c[i]))
            local_clearance = min(np.min(d) / 2.0, wall_dist)
            
            if local_clearance <= r_cur + tol:
                continue
            
            # Find direction to increase local clearance
            # Move away from closest constraint
            min_pair = np.argmin(d)
            min_wall_axis = np.argmin(0.5 - np.abs(c[i]))
            
            # Compute gradient of local clearance
            grad = np.zeros(2)
            
            # Pair contribution
            dij = c[i] - c[min_pair]
            norm = np.linalg.norm(dij)
            if norm > 1e-12:
                grad += dij / norm
            
            # Wall contribution
            wall_dists = 0.5 - np.abs(c[i])
            min_w = np.min(wall_dists)
            for k in range(2):
                if abs(wall_dists[k] - min_w) < 1e-12:
                    sign = np.sign(c[i, k])
                    if sign == 0:
                        sign = 1.0
                    grad[k] += -sign
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                continue
            grad /= grad_norm
            
            # Line search
            best_step = 0
            best_r_try = r_cur
            for step in np.linspace(0, 0.001, 15):
                c_try = c.copy()
                c_try[i] += step * grad
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > best_r_try:
                    best_r_try = r_try
                    best_step = step
            
            if best_step > 0:
                c[i] += best_step * grad
                c[i] = np.clip(c[i], -0.5 + 1e-12, 0.5 - 1e-12)
                r_cur = _radius_fast(c)
                improved = True
        
        if not improved:
            break
    
    return c, r_cur


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """
    SLSQP polish using epigraph formulation with active-set reduction.
    Variables: [x_1, y_1, ..., x_N, y_N, r]
    Maximize r subject to:
      - ||c_i - c_j|| >= 2r for all i < j
      - 0.5 - |c_ik| >= r for all i, k
    """
    from scipy.optimize import minimize as sp_minimize
    
    def objective(x):
        return -x[-1]  # minimize -r
    
    def objective_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g
    
    def _pair_constraint(x, i, j):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        return np.linalg.norm(c[i] - c[j]) - 2 * r
    
    def _wall_constraint(x, i, k):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        return 0.5 - np.abs(c[i, k]) - r
    
    def _pair_jac(x, i, j):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        g = np.zeros(2 * N + 1)
        dij = c[i] - c[j]
        norm = np.linalg.norm(dij)
        if norm > 1e-12:
            g[i*2] = dij[0] / norm
            g[i*2+1] = dij[1] / norm
            g[j*2] = -dij[0] / norm
            g[j*2+1] = -dij[1] / norm
        g[-1] = -2.0
        return g
    
    def _wall_jac(x, i, k):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        g = np.zeros(2 * N + 1)
        sign = np.sign(c[i, k])
        if sign == 0:
            sign = 1.0
        g[i*2 + k] = -sign
        g[-1] = -1.0
        return g
    
    x0 = np.concatenate([c_init.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    # First pass: find active set with full constraint set
    constraint_list = []
    for i in range(N):
        for j in range(i + 1, N):
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: _pair_constraint(x, i, j),
                'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
            })
    for i in range(N):
        for k in range(2):
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, i=i, k=k: _wall_constraint(x, i, k),
                'jac': lambda x, i=i, k=k: _wall_jac(x, i, k)
            })
    
    # Run SLSQP with the full constraint set first
    elapsed = time.time() - t0
    if elapsed < t0 + time_budget:
        try:
            res = sp_minimize(
                objective,
                best_x,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < objective(best_x):
                r_new = res.x[-1]
                c_new = res.x[:2*N].reshape(N, 2)
                r_check = _radius_fast(c_new)
                if r_check > best_r:
                    best_r = r_check
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Second pass: refine with active set only
    if time.time() - t0 < t0 + time_budget:
        active_pairs, active_walls = _kissing_graph(
            best_x[:2*N].reshape(N, 2), best_x[-1], tol=1e-3
        )
        
        if active_pairs or active_walls:
            active_constraint_list = []
            for (i, j) in active_pairs:
                active_constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: _pair_constraint(x, i, j),
                    'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
                })
            for (i, k) in active_walls:
                active_constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, k=k: _wall_constraint(x, i, k),
                    'jac': lambda x, i=i, k=k: _wall_jac(x, i, k)
                })
            
            try:
                res = sp_minimize(
                    objective,
                    best_x,
                    jac=objective_grad,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=active_constraint_list,
                    options={'maxiter': 1000, 'ftol': 1e-14}
                )
                if res.fun < objective(best_x):
                    r_new = res.x[-1]
                    c_new = res.x[:2*N].reshape(N, 2)
                    r_check = _radius_fast(c_new)
                    if r_check > best_r:
                        best_r = r_check
                        best_x = res.x.copy()
            except Exception:
                pass
    
    # Third pass: re-identify active set and refine again
    if time.time() - t0 < t0 + time_budget:
        active_pairs, active_walls = _kissing_graph(
            best_x[:2*N].reshape(N, 2), best_x[-1], tol=1e-4
        )
        
        if active_pairs or active_walls:
            active_constraint_list = []
            for (i, j) in active_pairs:
                active_constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: _pair_constraint(x, i, j),
                    'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
                })
            for (i, k) in active_walls:
                active_constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, k=k: _wall_constraint(x, i, k),
                    'jac': lambda x, i=i, k=k: _wall_jac(x, i, k)
                })
            
            try:
                res = sp_minimize(
                    objective,
                    best_x,
                    jac=objective_grad,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=active_constraint_list,
                    options={'maxiter': 2000, 'ftol': 1e-15}
                )
                if res.fun < objective(best_x):
                    r_new = res.x[-1]
                    c_new = res.x[:2*N].reshape(N, 2)
                    r_check = _radius_fast(c_new)
                    if r_check > best_r:
                        best_r = r_check
                        best_x = res.x.copy()
            except Exception:
                pass
    
    return best_x[:2*N].reshape(N, 2), best_r


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


def _active_set_refine(c, r, max_iter=100, tol=1e-10):
    """
    Active-set refinement: identify the kissing graph (pairs at distance ~2r
    and circles touching walls), then solve a local QP-like problem to
    maximize the minimum constraint while keeping active constraints tight.
    Uses a gradient-based approach on the active set.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for _ in range(max_iter):
        # Find active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall = 0.5 - np.abs(c)
        
        # Active pairs: distance close to 2*r
        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(d[i, j] - 2 * r_cur) < 1e-4:
                    active_pairs.append((i, j))
        
        # Active walls: circle close to wall
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall[i, k] < r_cur + 1e-4:
                    active_walls.append((i, k))
        
        if not active_pairs and not active_walls:
            break
        
        # Build gradient ascent direction to increase min radius
        # For each active constraint, push circles apart or away from wall
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
        
        # Normalize gradient
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12:
            break
        grad /= norm_grad
        
        # Line search
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
    in multiple directions, keeping the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 8 cardinal/diagonal + 16 fine angular
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in dirs:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
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
    while R <= R1 and (time.time() - t0) < DEADLINE - 20:
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
        if (time.time() - t0) > DEADLINE - 60:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 500, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish (smooth surrogate)
    if (time.time() - t0) < DEADLINE - 40:
        c_soft, r_soft = _lbfgs_soft_min_polish(
            best_c, best_r, t0,
            time_budget=min(25.0, DEADLINE - 40 - (time.time() - t0))
        )
        if r_soft > best_r:
            best_r = r_soft
            best_c = c_soft

    # Phase 3: SLSQP epigraph polish with active-set reduction
    if (time.time() - t0) < DEADLINE - 25:
        c_polished, r_polished = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(20.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 4: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=200)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 5: Active-set refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_as, r_as = _active_set_refine(best_c, best_r, max_iter=150)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 5:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=300, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 2:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=200, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c

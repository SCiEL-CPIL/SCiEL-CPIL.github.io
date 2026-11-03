"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 150s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _fast_radius(c, R):
    """Fast check: is the configuration valid at radius R? Returns min margin."""
    # Wall check
    wall = 0.5 - np.abs(c).max(axis=1)
    r_wall = wall.min()
    if r_wall < R:
        return r_wall - R
    # Pair check - use squared distances for speed
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = dist_sq.min()
    min_dist = np.sqrt(min_dist_sq)
    return min_dist / 2.0 - R

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    Uses fast radius check to reduce expensive _score calls.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    score_interval = 5  # only do full score every N steps
    step_count = 0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        step_count += 1
        # Fast check first
        fast_r = _fast_radius(c, R)
        if fast_r < -0.001:
            # Configuration is badly invalid, skip scoring
            R += step
            continue
        # Periodic full score
        if step_count % score_interval == 0 or R >= R1 - step:
            r = _score(c)
            if r > best[1]:
                best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    # Final score
    r = _score(c)
    if r > best[1]:
        best = (c.copy(), r)
    return best

def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _init_hex_tight():
    """Tighter hexagonal lattice."""
    positions = []
    spacing = 0.135
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(8):
            x = (col - 3.5) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_10x10_pick(rng):
    """10x10 grid, randomly pick 50."""
    ticks = (np.arange(10) + 0.5) / 10 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    return g[idx].copy()

def _init_clustered(rng):
    """Two clusters of 25 circles each."""
    c1 = rng.uniform(-0.4, -0.05, (25, 2))
    c2 = rng.uniform(0.05, 0.4, (25, 2))
    return np.vstack([c1, c2])

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def _soft_min_radius(c, beta=50.0):
    """Compute soft-min of all distances (pairs and walls)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ]), axis=0)
    
    # All distances
    all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], wall_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _soft_min_radius_and_grad(c, beta=50.0):
    """Compute soft-min radius and its gradient analytically (vectorized)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances (per circle, min of x and y wall distances)
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    
    # All distances: upper triangle pairs + wall distances
    # Use upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min value
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights
    weights = exp_vals / sum_exp
    
    grad = np.zeros((n, 2))
    
    # Pair distance gradients (vectorized)
    pair_weights = weights[:len(pair_dists)]
    # diff for upper triangle pairs
    pair_diff = diff[iu]  # shape (n_pairs, 2)
    pair_d = dists[iu]  # shape (n_pairs,)
    
    # direction = (c_i - c_j) / d
    safe_d = np.maximum(pair_d, 1e-12)
    directions = pair_diff / safe_d[:, np.newaxis]  # shape (n_pairs, 2)
    
    # Weighted sum: for each pair, w * direction contributes to grad[i] -= w*dir, grad[j] += w*dir
    weighted_dirs = pair_weights[:, np.newaxis] * directions  # shape (n_pairs, 2)
    
    # Scatter: for each pair (i,j), add -w*dir to grad[i] and +w*dir to grad[j]
    # Using np.add.at for accumulated additions
    np.add.at(grad, iu[0], -weighted_dirs)
    np.add.at(grad, iu[1], weighted_dirs)
    
    # Wall distance gradients (vectorized)
    wall_weights = weights[len(pair_dists):]
    # For each circle, determine which wall is limiting
    is_x_limit = wall_x <= wall_y
    # x contribution: sign depends on c[i,0]
    x_sign = np.sign(c[:, 0])
    x_sign[x_sign == 0] = 1.0
    # y contribution
    y_sign = np.sign(c[:, 1])
    y_sign[y_sign == 0] = 1.0
    
    # grad[i,0] += w * x_sign if x is limiting, else 0
    # grad[i,1] += w * y_sign if y is limiting, else 0
    grad[:, 0] += np.where(is_x_limit, wall_weights * x_sign, 0.0)
    grad[:, 1] += np.where(~is_x_limit, wall_weights * y_sign, 0.0)
    
    return soft_min, grad

def _scipy_polish(centers, deadline=None):
    """Polish using scipy with smooth soft-min objective and analytical gradient."""
    if deadline is not None and time.time() > deadline - 10.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    best_result = None
    best_r = _score(centers)

    for beta in [50.0, 100.0, 200.0, 400.0]:
        if deadline is not None and time.time() > deadline - 5.0:
            break

        def objective(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            return -_soft_min_radius(c, beta)

        def objective_grad(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, grad = _soft_min_radius_and_grad(c, beta)
            return -val, -grad

        bounds = [(-0.5, 0.5)] * (2 * n)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=objective_grad,
                              options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-15})
            polished = result.x.reshape(n, 2)
            r = _score(polished)
            if r > best_r:
                best_r = r
                best_result = polished
                x0 = result.x.copy()
        except Exception:
            pass

    if best_result is not None and best_r >= _score(centers) - 1e-10:
        return best_result

    return centers

def _contact_graph_refine(c, r, t0, deadline, max_iters=200):
    """
    Active-set refinement: identify circles in contact (distance <= 2r + tol)
    and solve for exact contact geometry using Newton's method on the
    constraint equations.
    """
    if deadline is not None and time.time() > deadline - 3.0:
        return c
    
    c = c.copy()
    n = len(c)
    tol = 1e-8
    
    for outer_iter in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Identify contact pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Contact pairs: distance within tolerance of 2r
        contact_mask = dist < (2 * r + tol)
        np.fill_diagonal(contact_mask, False)
        
        # Wall contacts
        wall_contact = np.zeros(n, dtype=bool)
        wall_contact |= (0.5 - np.abs(c[:, 0])) < (r + tol)
        wall_contact |= (0.5 - np.abs(c[:, 1])) < (r + tol)
        
        # Build the system of equations for contact pairs
        # For each contact pair (i,j): ||c_i - c_j||^2 = (2r)^2
        # For each wall contact i: c_i on boundary
        
        # Use simple iterative correction
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if contact_mask[i, j]:
                    d = c[i] - c[j]
                    dist_ij = np.linalg.norm(d)
                    if dist_ij > 1e-12:
                        # Push apart to exactly 2r
                        target = 2 * r
                        scale = target / dist_ij
                        correction = (scale - 1) * d * 0.5
                        c[i] += correction
                        c[j] -= correction
                        improved = True
        
        # Wall corrections
        for i in range(n):
            if wall_contact[i]:
                for axis in range(2):
                    if c[i, axis] > 0.5 - r:
                        c[i, axis] = 0.5 - r
                        improved = True
                    elif c[i, axis] < -0.5 + r:
                        c[i, axis] = -0.5 + r
                        improved = True
        
        if not improved:
            break
        
        c = np.clip(c, -0.5, 0.5)
    
    return c

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
        _init_hex_tight(),
        _init_grid_10x10_pick(np.random.default_rng(123)),
        _init_clustered(np.random.default_rng(456)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 35) / n_inits
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0003, 1200, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        if time.time() - t0 > DEADLINE - 35:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Langevin refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=10000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: Analytic gradient polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_polished = _scipy_polish(best, DEADLINE)
        r_polished = _score(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best = c_polished
    
    # Phase 4: Contact graph refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 6:
        c_contact = _contact_graph_refine(best, best_r, t0, DEADLINE)
        r_contact = _score(c_contact)
        if r_contact > best_r:
            best_r = r_contact
            best = c_contact
    
    # Phase 5: Fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 6:
        r_cur = _score(best)
        c = _relax(best, r_cur, 4000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Phase 6: Second Langevin pass with lower temperature
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=8000, T0=0.0008, T1=0.00002, lr=0.002)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 7: Final analytic polish with tighter settings
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        c_polished = _scipy_polish(best, DEADLINE)
        r_polished = _score(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best = c_polished
    
    # Phase 8: Final micro-relaxation
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 3:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.005, 0.0002, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best

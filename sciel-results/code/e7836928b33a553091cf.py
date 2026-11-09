"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin before 120s kill


def _pairwise_min_dist(c):
    """Compute minimum pairwise distance using matrix ops (faster than pdist for N=50)."""
    # c has shape (N, 2)
    # Compute all pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 2)
    sq_dists = np.sum(diff * diff, axis=2)  # (N, N)
    # Set diagonal to inf to exclude self-distances
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = sq_dists.min()
    return np.sqrt(min_sq)


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances
    min_pair = _pairwise_min_dist(c)
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _smooth_score(c_flat, beta=50.0):
    """Smooth surrogate for the radius score using log-sum-exp."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances via matrix
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    d = np.sqrt(sq_dists)
    # Only use upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d_upper = d[iu]
    
    if d_upper.size == 0:
        return 0.0
    
    # Wall distances for each point
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    # Soft-min of pairwise distances (use half-distance since r = min_pair/2)
    d_half = d_upper / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    # Soft-min of wall distances
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    # Combine
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat, beta=50.0):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat, beta)


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    rows = int(np.ceil(N / 8))
    cols = 8
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2) * 0.22
            y = (i - (rows - 1) / 2) * 0.2
            if i % 2 == 1:
                x += 0.11
            points.append([x, y])
    return np.array(points[:N])


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _local_refine(c, t0, deadline=DEADLINE - 10):
    """Local refinement using L-BFGS-B with adaptive beta."""
    best_c = c.copy()
    best_s = _score_from_centres(best_c)
    x0 = best_c.ravel()
    
    for beta in [30.0, 60.0, 120.0, 250.0]:
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize(
                lambda cf, b=beta: _negative_score(cf, b), x0,
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            c_new = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s_new = _score_from_centres(c_new)
            if s_new > best_s:
                best_s = s_new
                best_c = c_new
            x0 = c_new.ravel()
        except Exception:
            pass
    
    return best_c, best_s


def _sa_optimize(c0, t0, deadline=DEADLINE - 10, rng_sa=None):
    """Simulated annealing with local refinement."""
    if rng_sa is None:
        rng_sa = np.random.default_rng(42)
    
    c = c0.copy()
    current_score = _score_from_centres(c)
    best_c = c.copy()
    best_s = current_score
    
    n_iter = 8000
    t_max = 0.015
    t_min = 1e-6
    
    for it in range(n_iter):
        if time.time() - t0 > deadline:
            break
        
        # Temperature schedule
        T = t_max * (t_min / t_max) ** (it / n_iter)
        
        # Generate move: perturb a random subset of points
        n_move = rng_sa.integers(1, 8)
        idx = rng_sa.choice(N, size=n_move, replace=False)
        
        c_new = c.copy()
        # Scale perturbation with temperature
        scale = 0.015 + 0.10 * (T / t_max)
        delta = rng_sa.normal(0, scale, size=(n_move, 2))
        c_new[idx] += delta
        c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
        
        new_score = _score_from_centres(c_new)
        delta_s = new_score - current_score
        
        # Accept if better or with probability exp(-delta_s/T) for worse
        if delta_s > 0 or (T > 0 and rng_sa.random() < np.exp(max(-100, delta_s / T))):
            c = c_new
            current_score = new_score
            if current_score > best_s:
                best_s = current_score
                best_c = c.copy()
        
        # Every 40 iterations, do a local refinement on best
        if it % 40 == 0 and it > 0:
            best_c, best_s = _local_refine(best_c, t0, deadline)
            # Update current to best to avoid getting stuck
            c = best_c.copy()
            current_score = best_s
    
    return best_c, best_s


def construct():
    t0 = time.time()
    
    # Try multiple initial configurations
    inits = [_grid_init(), _hex_init()]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)

    # Strategy 1: Local optimisation from multiple starts with smooth objective
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    n_starts = 20
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        elif i == 2:
            x0 = _grid_init().ravel()
        elif i == 3:
            # Slightly perturbed hex
            x0 = _hex_init().ravel()
            x0 = x0 + rng.normal(0, 0.02, size=x0.shape)
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)

        try:
            c_init = x0.reshape(N, 2)
            c_init = np.clip(c_init, -HALF + 1e-9, HALF - 1e-9)
            c_refined, s_refined = _local_refine(c_init, t0, DEADLINE - 30)
            if s_refined > best_score:
                best_score = s_refined
                best_centres = c_refined
        except Exception:
            pass

    # Strategy 2: Simulated annealing from best configuration
    if time.time() - t0 < DEADLINE - 20:
        best_centres, best_score = _sa_optimize(best_centres, t0, DEADLINE - 5, rng)
    
    # Final refinement
    if time.time() - t0 < DEADLINE - 2:
        best_centres, best_score = _local_refine(best_centres, t0, DEADLINE - 1)
    
    return best_centres

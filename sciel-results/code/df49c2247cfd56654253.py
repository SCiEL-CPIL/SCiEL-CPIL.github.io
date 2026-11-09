"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin before 120s kill


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
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
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
    """Smooth surrogate for the radius score using log-sum-exp.
    
    Instead of min(distances), we use a soft-min that is differentiable.
    This allows gradient-based optimizers to navigate the landscape smoothly.
    """
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances
    d = pdist(c)
    if d.size == 0:
        return 0.0
    
    # Wall distances for each point
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    # Soft-min of pairwise distances (each pair contributes distance/2 to radius)
    # Use log-sum-exp trick for numerical stability
    d_half = d / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    # Soft-min of wall distances
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    # Combine: we want min(soft_min_pair, soft_min_wall)
    # Use another soft-min
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat)


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    # Hexagonal close packing for better initial configuration
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


def construct():
    t0 = time.time()
    
    # Try multiple initial configurations
    inits = [_grid_init(), _hex_init()]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)

    # Strategy 1: Local optimisation from multiple starts with smooth objective
    # Start with the better initial config
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    n_starts = 15
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 25:
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        elif i == 2:
            x0 = _grid_init().ravel()
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)

        try:
            # Use smooth objective for gradient-based optimization
            res = minimize(
                _negative_score, x0,
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-9, 'iprint': 0}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Strategy 2: Simulated annealing with targeted local moves
    # This explores the global landscape more effectively than DE for this problem
    if time.time() - t0 < DEADLINE - 15:
        def _sa_optimize(c0, t_max=0.01, t_min=1e-5, n_iter=3000, rng_sa=None):
            """Simulated annealing with local refinement."""
            if rng_sa is None:
                rng_sa = np.random.default_rng()
            
            c = c0.copy()
            current_score = _score_from_centres(c)
            best_c = c.copy()
            best_s = current_score
            
            for it in range(n_iter):
                if time.time() - t0 > DEADLINE - 12:
                    break
                
                # Temperature schedule
                T = t_max * (t_min / t_max) ** (it / n_iter)
                
                # Generate move: perturb a random subset of points
                n_move = rng_sa.integers(1, 5)
                idx = rng_sa.choice(N, size=n_move, replace=False)
                
                c_new = c.copy()
                # Random perturbation
                delta = rng_sa.normal(0, 0.05, size=(n_move, 2))
                c_new[idx] += delta
                c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
                
                new_score = _score_from_centres(c_new)
                delta_s = new_score - current_score
                
                # Accept if better or with probability exp(-delta_s/T) for worse
                if delta_s > 0 or rng_sa.random() < np.exp(delta_s / T):
                    c = c_new
                    current_score = new_score
                    if current_score > best_s:
                        best_s = current_score
                        best_c = c.copy()
                
                # Every 50 iterations, do a local refinement on best
                if it % 50 == 0 and it > 0:
                    try:
                        res = minimize(
                            _negative_score, best_c.ravel(),
                            method='L-BFGS-B',
                            bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                            options={'maxiter': 100, 'ftol': 1e-12, 'iprint': 0}
                        )
                        c_refined = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                        s_refined = _score_from_centres(c_refined)
                        if s_refined > best_s:
                            best_s = s_refined
                            best_c = c_refined
                            c = best_c.copy()
                            current_score = best_s
                    except Exception:
                        pass
            
            return best_c, best_s
        
        sa_c, sa_s = _sa_optimize(best_centres, t_max=0.015, t_min=1e-6, n_iter=4000, rng_sa=rng)
        if sa_s > best_score:
            best_score = sa_s
            best_centres = sa_c

    # Strategy 3: High-precision polish with tight tolerances
    if time.time() - t0 < DEADLINE - 5:
        try:
            # Multiple polish passes with decreasing tolerance
            for ftol_val in [1e-12, 1e-14, 1e-16]:
                res = minimize(
                    _negative_score, best_centres.ravel(),
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 2000, 'ftol': ftol_val, 'gtol': 1e-12, 'iprint': 0}
                )
                c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 4: Final exact local solve on constraining pairs
    if time.time() - t0 < DEADLINE - 2:
        try:
            # Identify the most constraining pairs and do targeted optimization
            c = best_centres
            d = pdist(c)
            min_pair = d.min()
            min_wall = np.min(np.stack([
                c[:, 0] + HALF,
                HALF - c[:, 0],
                c[:, 1] + HALF,
                HALF - c[:, 1]
            ]), axis=0).min()
            r_current = min(min_pair / 2.0, min_wall)
            
            # Try SLSQP with very tight tolerance
            res = minimize(
                _negative_score, c.ravel(),
                method='SLSQP',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'iprint': 0}
            )
            c_final = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s_final = _score_from_centres(c_final)
            if s_final > best_score:
                best_score = s_final
                best_centres = c_final
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres

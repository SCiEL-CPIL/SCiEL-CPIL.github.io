import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score_centres(centres):
    """
    Compute the packing radius for a set of centres.
    Returns the minimum of:
    1. Distance to the nearest wall for each sphere.
    2. Half the distance to the nearest other sphere for each pair.
    """
    if len(centres) == 0:
        return 0.0
    
    # Wall distances: min_i min_k (0.5 - |c_ik|)
    # max abs coord per sphere
    max_abs = np.max(np.abs(centres), axis=1)
    wall_dists = 0.5 - max_abs
    r_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for efficiency, though for N=50 it's small
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_score(centres, beta=100.0):
    """
    Smooth surrogate for the packing radius.
    r = min(constraints)
    soft_min = -1/beta * log(sum(exp(-beta * constraints)))
    """
    n = len(centres)
    if n == 0:
        return 0.0
    
    # Wall constraints
    max_abs = np.max(np.abs(centres), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Pairwise constraints (distance / 2)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices for pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    all_vals = np.concatenate([wall_dists, pair_dists])
    
    # Soft min calculation
    # To avoid overflow, subtract max
    max_val = np.max(all_vals)
    if max_val < -1e10:
        return 0.0
        
    # min(x) = -max(-x)
    # max(-x) approx (1/beta) log(sum(exp(-beta x)))
    neg_vals = -beta * all_vals
    max_neg = np.max(neg_vals)
    
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
    soft_min = -log_sum_exp / beta
    
    return float(soft_min)

def _random_start(n, seed=42):
    """
    Initialize with a grid structure to avoid random clustering.
    """
    rng = np.random.RandomState(seed)
    # Approximate cubic root
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Ensure we have at least n points
    while m**3 < n:
        m += 1
        
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # If grid is larger than n, take first n
    if len(grid) > n:
        grid = grid[:n]
    elif len(grid) < n:
        # Pad with random points (should not happen with m**3 >= n)
        extra = rng.uniform(-0.4, 0.4, size=(n - len(grid), 3))
        grid = np.vstack([grid, extra])
        
    # Add small noise to break symmetry
    noise = rng.normal(0, 0.02, size=grid.shape)
    grid = grid + noise
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _local_optimize_smooth(centres, beta=100.0, time_limit=10.0):
    """
    Optimize using L-BFGS-B on the soft-min surrogate.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 3)
        # Penalize if outside bounds significantly, though bounds are set
        if np.any(np.abs(c) > 0.5):
            return 1e10
        return -_soft_min_score(c, beta=beta)
    
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(objective, centres.flatten(), method='L-BFGS-B',
                          bounds=bounds, 
                          options={'maxiter': 500, 'maxfun': 1000, 
                                   'ftol': 1e-12, 'gtol': 1e-8,
                                   'disp': False})
        if result.success or result.fun < 0:
            return result.x.reshape(-1, 3)
    except (TimeoutError, Exception):
        pass
        
    return centres

def _high_precision_polish(centres, time_limit=10.0):
    """
    Refine the solution by identifying the most constrained spheres and moving them apart.
    This is a direct gradient ascent on the exact (non-smooth) objective.
    """
    start_time = time.time()
    current = centres.copy()
    current_score = _score_centres(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.005
    n_iter = 0
    max_iter = 2000
    
    while time.time() - start_time < time_limit and n_iter < max_iter:
        n_iter += 1
        c = current
        n = len(c)
        
        # Calculate constraints
        max_abs = np.max(np.abs(c), axis=1)
        wall_dists = 0.5 - max_abs
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1)) / 2.0
        np.fill_diagonal(dists, np.inf)
        
        # Find global minimum constraint
        min_wall = np.min(wall_dists)
        min_pair = np.min(dists)
        
        if min_wall < min_pair:
            # Most constrained by wall
            idxs = np.where(wall_dists <= min_wall + 1e-10)[0]
            if len(idxs) == 0: continue
            idx = idxs[0]
            
            # Move away from wall
            # Determine which wall
            for axis in range(3):
                if c[idx, axis] > 0:
                    c[idx, axis] -= step_size
                else:
                    c[idx, axis] += step_size
        else:
            # Most constrained by pair
            # Find the pair with minimum distance
            min_val = min_pair
            # Get indices
            iu = np.triu_indices(n, k=1)
            flat_dists = dists[iu]
            min_flat_idx = np.argmin(flat_dists)
            i, j = iu[0][min_flat_idx], iu[1][min_flat_idx]
            
            # Move i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                c[i] += direction * step_size * 0.5
                c[j] -= direction * step_size * 0.5
            else:
                # If coincident, random nudge
                c[i] += np.random.normal(0, step_size, size=3)
                c[j] -= np.random.normal(0, step_size, size=3)
        
        c = np.clip(c, -0.5, 0.5)
        
        new_score = _score_centres(c)
        
        if new_score > current_score:
            current = c
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
            # Increase step size slightly if improving
            step_size = min(step_size * 1.05, 0.02)
        else:
            # Decrease step size if not improving
            step_size *= 0.9
            if step_size < 1e-6:
                break
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_centres = None
    best_score = -1.0
    
    # 1. Multiple Random Starts with SA
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
            
        c = _random_start(N, seed=seed)
        
        # Quick SA to refine start
        c = _simulated_annealing(c, time_limit=5.0, iters=2000)
        
        s = _score_centres(c)
        if s > best_score:
            best_centres = c
            best_score = s
            
    # 2. L-BFGS-B Refinement on best start
    if time.time() < deadline - 20 and best_centres is not None:
        opt_centres = _local_optimize_smooth(best_centres, beta=100.0, time_limit=15.0)
        opt_score = _score_centres(opt_centres)
        if opt_score > best_score:
            best_centres = opt_centres
            best_score = opt_score
            
    # 3. High Precision Polish
    if time.time() < deadline - 10 and best_centres is not None:
        polished = _high_precision_polish(best_centres, time_limit=10.0)
        polished_score = _score_centres(polished)
        if polished_score > best_score:
            best_centres = polished
            best_score = polished_score
            
    # 4. Final L-BFGS-B with higher beta (sharper min)
    if time.time() < deadline - 5 and best_centres is not None:
        final_opt = _local_optimize_smooth(best_centres, beta=200.0, time_limit=5.0)
        final_score = _score_centres(final_opt)
        if final_score > best_score:
            best_centres = final_opt
            best_score = final_score
            
    # 5. Final Polish
    if time.time() < deadline - 2 and best_centres is not None:
        final_polish = _high_precision_polish(best_centres, time_limit=3.0)
        final_polish_score = _score_centres(final_polish)
        if final_polish_score > best_score:
            best_centres = final_polish
            best_score = final_polish_score
            
    if best_centres is None:
        # Fallback
        best_centres = _random_start(N, seed=0)
        
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres

def _simulated_annealing(centres, time_limit=10.0, iters=5000):
    """
    Simulated Annealing with vectorized scoring.
    """
    start_time = time.time()
    current = centres.copy()
    current_score = _score_centres(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.02
    T_min = 1e-6
    cool = 0.999
    
    rng = np.random.RandomState(42)
    
    for it in range(iters):
        if time.time() - start_time > time_limit:
            break
        if T < T_min:
            break
            
        candidate = current.copy()
        
        # Move strategy: Perturb a random sphere
        idx = rng.randint(N)
        # Scale with T
        delta = rng.normal(0, T, size=3)
        candidate[idx] += delta
        candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
        
        cand_score = _score_centres(candidate)
        delta_score = cand_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
                
        T *= cool
        
    return best

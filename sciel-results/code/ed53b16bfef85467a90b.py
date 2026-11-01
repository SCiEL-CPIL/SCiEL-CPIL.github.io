"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def compute_score(centers):
    """Compute the packing score: minimum radius."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dist = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_wall, min_pair / 2.0)


def init_grid(perturb=0.01, seed=0):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    grid = grid + rng.uniform(-perturb, perturb, size=grid.shape)
    return np.clip(grid, -0.45, 0.45)


def init_random(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def soft_min_packing(x, beta):
    """
    Objective: negative soft-min of all constraints.
    x: (N, 3) array of centers
    beta: temperature for log-sum-exp
    Returns: -soft_min(constraints) where constraints are:
      - pairwise distances / 2
      - wall distances
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    pair_dist = np.sqrt(dist2) / 2.0
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centers)
    wall_flat = wall_dist.flatten()
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_dist.flatten(), wall_flat])
    all_constraints = all_constraints[all_constraints < np.inf]
    
    if len(all_constraints) == 0:
        return 0.0
    
    # Log-sum-exp approximation to min
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    log_sum = min_val + np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return -log_sum


def soft_min_packing_grad(x, beta):
    """Gradient of soft_min_packing."""
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    pair_dist = np.sqrt(dist2) / 2.0
    
    # Wall distances
    wall_dist = 0.5 - np.abs(centers)
    wall_flat = wall_dist.flatten()
    
    all_constraints = np.concatenate([pair_dist.flatten(), wall_flat])
    all_constraints = all_constraints[all_constraints < np.inf]
    
    if len(all_constraints) == 0:
        return np.zeros_like(x)
    
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient with respect to centers
    grad = np.zeros((N, 3))
    
    # Pairwise contribution
    n_pairs = pair_dist.size
    pair_weights = weights[:n_pairs]
    
    # Reconstruct pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = pair_weights[k]
        if w > 1e-15:
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d) + 1e-15
            # d(pair_dist)/d(centers[i]) = d/dist * 0.5
            grad[i] += -w * d / dist * 0.5
            grad[j] += -w * (-d) / dist * 0.5
    
    # Wall contribution
    wall_weights = weights[n_pairs:]
    wall_reshaped = wall_weights.reshape(N, 3)
    signs = np.sign(centers)
    signs[signs == 0] = 1.0
    grad += -wall_reshaped * signs
    
    return grad


def l_bfgs_b_optimize(initial, beta_start=50, beta_end=500, n_beta=5, max_iter_per_beta=100):
    """Run L-BFGS-B with increasing beta (decreasing temperature)."""
    x = initial.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    for bi in range(n_beta):
        frac = bi / max(n_beta - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        res = minimize(
            soft_min_packing,
            x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter_per_beta, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x = res.x
    
    centers = x.reshape(N, 3)
    return centers


def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initializations
    attempts = []
    for seed in range(4):
        attempts.append(init_grid(perturb=0.01, seed=seed))
    for seed in range(4):
        attempts.append(init_random(seed=seed))
    
    # Additional structured grids
    for (m1, m2, m3) in [(5, 5, 4), (5, 4, 5), (4, 5, 5)]:
        t1 = (np.arange(m1) + 0.5) / m1 - 0.5
        t2 = (np.arange(m2) + 0.5) / m2 - 0.5
        t3 = (np.arange(m3) + 0.5) / m3 - 0.5
        g = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])[:N]
        rng = np.random.RandomState(300 + m1)
        g = g + rng.uniform(-0.005, 0.005, size=g.shape)
        g = np.clip(g, -0.45, 0.45)
        attempts.append(g)
    
    for idx, initial in enumerate(attempts):
        if time.time() > deadline - 15:
            break
        
        try:
            optimized = l_bfgs_b_optimize(initial, beta_start=30, beta_end=400, n_beta=4, max_iter_per_beta=80)
            score = compute_score(optimized)
            
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
        except Exception:
            continue
    
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
        return grid
    
    return best_centers
# EVOLVE-BLOCK-END

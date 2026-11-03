import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    if r_pair < r_wall:
        return r_pair, 'pair', min_pair_idx, r_pair, r_wall
    else:
        return r_wall, 'wall', min_wall_idx, r_pair, r_wall

def soft_min_objective(flat_pos, beta=50.0):
    """
    Smooth surrogate objective using soft-min.
    We want to maximize r = min(r_pair, r_wall).
    Instead, we minimize -soft_min(r_pair, r_wall) where soft_min is a smooth approximation.
    
    soft_min(a, b) = -log(exp(-beta*a) + exp(-beta*b)) / beta
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Collect all radius candidates
    # r_pair values: dists / 2 for all pairs
    # r_wall values: d_walls for all (i, k)
    
    # For efficiency, we'll use a log-sum-exp style soft-min
    # r = min(r_pair_min, r_wall_min)
    # soft_r = -log(exp(-beta*r_pair_min) + exp(-beta*r_wall_min)) / beta
    
    r_pair_min = np.min(dists) / 2.0
    r_wall_min = np.min(d_walls)
    
    # Soft-min of two values
    # To avoid overflow, use the stable form
    m = min(r_pair_min, r_wall_min)
    if m < -1000:
        return r_pair_min, None  # Fallback to hard min
    
    exp_pair = np.exp(-beta * (r_pair_min - m))
    exp_wall = np.exp(-beta * (r_wall_min - m))
    
    soft_r = m - np.log(exp_pair + exp_wall) / beta
    
    # Gradient computation
    # d(soft_r)/d(pos) = (1/beta) * (exp_pair * d(r_pair_min)/d(pos) * (-beta) + exp_wall * d(r_wall_min)/d(pos) * (-beta)) / (exp_pair + exp_wall)
    # = -(exp_pair * grad_r_pair + exp_wall * grad_r_wall) / (exp_pair + exp_wall)
    
    # Compute gradient of r_pair_min
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    d = dists[i, j]
    
    grad_r_pair = np.zeros_like(pos)
    if d > 1e-12:
        vec = (pos[i] - pos[j]) / d
        grad_r_pair[i] += 0.5 * vec
        grad_r_pair[j] += -0.5 * vec
    
    # Compute gradient of r_wall_min
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    grad_r_wall = np.zeros_like(pos)
    s = np.sign(pos[wi, wk])
    if s == 0:
        s = 1.0
    grad_r_wall[wi, wk] = -s
    
    # Combine
    total_exp = exp_pair + exp_wall
    if total_exp < 1e-300:
        return soft_r, None
    
    grad_soft_r = -(exp_pair * grad_r_pair + exp_wall * grad_r_wall) / total_exp
    
    # We want to minimize -soft_r, so objective = -soft_r, grad = -grad_soft_r
    obj = -soft_r
    grad = -grad_soft_r.flatten()
    
    return obj, grad

def objective_and_grad(flat_pos):
    """Original hard-min objective with analytical gradient."""
    pos = flat_pos.reshape(N, 3)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    grad = np.zeros_like(pos)
    
    if r_pair <= r_wall:
        d = min_pair_val
        if d > 1e-12:
            vec = (pos[i] - pos[j]) / d
            grad[i] += 0.5 * vec
            grad[j] += -0.5 * vec
        return -r_pair, -grad.flatten()
    else:
        s = np.sign(pos[wi, wk])
        if s == 0:
            s = 1.0
        grad[wi, wk] = -s
        return -r_wall, -grad.flatten()

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass using the hard-min objective."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            objective_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Better grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    # Evaluate initial
    val, _, _, _, _ = compute_min_dist(best_pos)
    best_r = val
    
    rng = np.random.default_rng(42)
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Decide start point with adaptive perturbation
        if attempts % 5 == 0:
            # Random start
            start_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        elif attempts % 5 == 1:
            # Small perturbation of best
            scale = 0.02
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 2:
            # Medium perturbation
            scale = 0.05
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 3:
            # Large perturbation
            scale = 0.1
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        else:
            # Very small perturbation for fine-tuning
            scale = 0.005
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        
        # Clip to stay inside bounds
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Try both soft-min and hard-min objectives
        use_soft = (attempts % 2 == 0)
        
        try:
            if use_soft:
                res = minimize(
                    lambda x: soft_min_objective(x, beta=50.0)[0],
                    start_pos.flatten(),
                    jac=lambda x: soft_min_objective(x, beta=50.0)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            else:
                res = minimize(
                    objective_and_grad,
                    start_pos.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            
            if res.success or res.fun < -best_r:
                new_pos = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_pos)):
                    val, _, _, _, _ = compute_min_dist(new_pos)
                    if val > best_r:
                        best_r = val
                        best_pos = new_pos
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=300, ftol=1e-12, gtol=1e-10)
        val, _, _, _, _ = compute_min_dist(polished_pos)
        if val > best_r:
            best_r = val
            best_pos = polished_pos
    
    return best_pos

# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    # pos is (N, 3)
    # Calculate pairwise distances
    # dist[i, j] = ||pos[i] - pos[j]||
    # We want min_{i<j} dist[i,j] / 2 and min_i min_k (HALF - |pos[i,k]|)
    
    # Pairwise
    # Use broadcasting
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair = np.min(dists) / 2.0
    
    # Wall
    # d_wall = np.min(HALF - np.abs(pos), axis=1)
    # min_wall = np.min(d_wall)
    
    # return min(min_pair, min_wall)
    
    # For speed and gradient, we need indices
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    
    # Compare
    # Note: min_pair_val is distance between centers, radius is min_pair_val / 2
    # min_wall_val is distance to wall, radius is min_wall_val
    # We want to maximize r.
    # Let's work with 2*r to avoid division? No, just keep consistent.
    # Let's maximize r.
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    if r_pair < r_wall:
        # Active constraint is pair
        return r_pair, 'pair', min_pair_idx, r_pair, r_wall
    else:
        # Active constraint is wall
        return r_wall, 'wall', min_wall_idx, r_pair, r_wall

def objective_and_grad(flat_pos):
    pos = flat_pos.reshape(N, 3)
    
    # Calculate distances
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
    
    # We want to MINIMIZE -r
    # So objective is -min(r_pair, r_wall)
    
    grad = np.zeros_like(pos)
    
    if r_pair <= r_wall:
        # Active is pair i, j
        # r = 0.5 * ||pos[i] - pos[j]||
        # dr/dpos[i] = 0.5 * (pos[i] - pos[j]) / ||pos[i] - pos[j]||
        # dr/dpos[j] = 0.5 * (pos[j] - pos[i]) / ||pos[i] - pos[j]||
        d = min_pair_val
        if d > 1e-12:
            vec = (pos[i] - pos[j]) / d
            grad[i] += 0.5 * vec
            grad[j] += -0.5 * vec
        # Objective is -r, so grad_obj = -grad_r
        return -r_pair, -grad.flatten()
    else:
        # Active is wall for sphere wi, axis wk
        # r = HALF - |pos[wi, wk]|
        # dr/dpos[wi, wk] = -sign(pos[wi, wk])
        s = np.sign(pos[wi, wk])
        if s == 0:
            s = 1.0 # arbitrary
        grad[wi, wk] = -s
        # Objective is -r
        return -r_wall, -grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    # Evaluate initial
    val, _, _, _, _ = compute_min_dist(best_pos)
    best_r = val
    
    # If grid is not full N, pad with random or just use N=100 grid logic?
    # 4^3 = 64. 5^3 = 125.
    # If m=5, we have 125 points, take 100.
    # This is a valid start.
    
    rng = np.random.default_rng(42)
    
    # Basin Hopping / Multi-start
    # We will try to optimize the current best, then perturb it.
    # Also try random starts.
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Decide start point
        if attempts % 3 == 0:
            # Random start
            start_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
            # Perturb best
            # Perturbation scale should decrease? Or stay constant?
            # Let's use a moderate perturbation
            scale = 0.05
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
            # Clip to stay inside? Not strictly necessary if optimizer handles bounds, 
            # but our objective function doesn't have bounds in scipy call.
            # We should use bounds in minimize.
            
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run Optimization
        try:
            res = minimize(
                objective_and_grad, 
                start_pos.flatten(), 
                jac=True, # We return grad
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < -best_r:
                new_pos = res.x.reshape(N, 3)
                # Validate and compute score
                # Ensure finite
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
            
    return best_pos

# EVOLVE-BLOCK-END

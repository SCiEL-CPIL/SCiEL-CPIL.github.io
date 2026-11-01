"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum radius for a given set of centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    # Centers are in [-0.5, 0.5]
    wall_dists = np.concatenate([
        0.5 - centers[:, 0], 
        0.5 + centers[:, 0],
        0.5 - centers[:, 1], 
        0.5 + centers[:, 1]
    ])
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _objective(x):
    """Objective function: Minimize sum of inverse distances (pairwise + wall)."""
    centers = x.reshape(-1, 2)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, 1)
    pair_dist = dist[iu]
    # Avoid division by zero if points coincide (should not happen with jitter)
    pair_dist = np.maximum(pair_dist, 1e-6)
    pair_energy = np.sum(1.0 / pair_dist)
    
    # Wall
    wall_dists = np.concatenate([
        0.5 - centers[:, 0], 
        0.5 + centers[:, 0],
        0.5 - centers[:, 1], 
        0.5 + centers[:, 1]
    ])
    wall_dists = np.maximum(wall_dists, 1e-6)
    wall_energy = np.sum(1.0 / wall_dists)
    
    return pair_energy + wall_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_centers = None
    best_radius = -1.0
    
    # Try multiple random initializations
    seeds = [42, 123, 456, 789, 1000]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Initialization: Jittered Grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        
        # Jitter
        jitter_scale = 0.02
        x0 = grid + np.random.uniform(-jitter_scale, jitter_scale, grid.shape)
        x0 = np.clip(x0, -0.49, 0.49)
        
        # Optimize
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            x_opt = res.x.reshape(-1, 2)
            
            # Ensure validity (clip to square)
            x_opt = np.clip(x_opt, -0.5, 0.5)
            
            r = _compute_radius(x_opt)
            
            if r > best_radius:
                best_radius = r
                best_centers = x_opt.copy()
                
        except Exception:
            continue
            
    # If we have time, do a local polish on the best result
    if best_centers is not None and time.time() < deadline - 10.0:
        # Simple local polish: iterate on closest pairs
        centers = best_centers.copy()
        for _ in range(50): # Max polish iterations
            if time.time() > deadline:
                break
                
            # Find closest pair
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
            i, j = min_idx
            
            # Check wall constraints too
            wall_dists = np.concatenate([
                0.5 - centers[:, 0], 
                0.5 + centers[:, 0],
                0.5 - centers[:, 1], 
                0.5 + centers[:, 1]
            ])
            min_wall_idx = np.argmin(wall_dists)
            
            # Determine which constraint is tighter
            pair_dist = np.sqrt(dist_sq[i, j])
            wall_dist = wall_dists[min_wall_idx]
            
            # We want to increase the minimum of (pair_dist/2, wall_dist)
            # If pair is tighter, push them apart. If wall is tighter, push inwards.
            # This is complex to do perfectly, so we'll just do a small random perturbation 
            # and accept if radius improves.
            
            # Simple random perturbation polish
            old_r = _compute_radius(centers)
            
            # Perturb all points slightly
            noise = np.random.normal(0, 0.001, centers.shape)
            new_centers = centers + noise
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_r = _compute_radius(new_centers)
            
            if new_r > old_r:
                centers = new_centers
            else:
                # Try smaller step
                noise = np.random.normal(0, 0.0001, centers.shape)
                new_centers = centers + noise
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r = _compute_radius(new_centers)
                if new_r > old_r:
                    centers = new_centers
                else:
                    break # No improvement, stop
            
            best_radius = _compute_radius(centers)
            best_centers = centers.copy()

    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        return grid

    return best_centers

# EVOLVE-BLOCK-END

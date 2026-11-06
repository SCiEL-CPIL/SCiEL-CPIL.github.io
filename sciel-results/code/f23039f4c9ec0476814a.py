"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_radius(pts):
    """Calculate the maximum radius for a given set of points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distances to walls
    # Square is [-0.5, 0.5]
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    min_wall = np.min(dists_wall)
    
    # Radius is limited by half the minimum pair distance OR the minimum wall distance
    return min(min_pair / 2.0, min_wall)

def _energy(x):
    """
    Energy function to minimize.
    Uses inverse square repulsion to spread points out.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero on diagonal
    
    # Wall distances
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    
    # Clip to avoid infinity
    dists = np.maximum(dists, 1e-6)
    dists_wall = np.maximum(dists_wall, 1e-6)
    
    e_pairs = np.sum(1.0 / (dists ** 2))
    e_walls = np.sum(1.0 / (dists_wall ** 2))
    
    return e_pairs + e_walls

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    best_x = None
    best_r = 0.0
    
    # Number of restarts
    n_restarts = 30
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with grid + jitter
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        x0 = grid[:N].copy()
        
        # Random jitter to break symmetry
        np.random.seed(i + 42)
        x0 += np.random.uniform(-0.02, 0.02, x0.shape)
        
        # Ensure initial points are strictly inside bounds to help optimizer
        x0 = np.clip(x0, -0.49, 0.49)
        
        try:
            res = minimize(_energy, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            x_opt = res.x.reshape(N, 2)
            
            r = _calc_radius(x_opt)
            
            if r > best_r:
                best_r = r
                best_x = x_opt.copy()
                
        except Exception:
            pass

    if best_x is None:
        # Fallback: return grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_x = np.array([(x, y) for y in ticks for x in ticks])[:N]
        
    return best_x
# EVOLVE-BLOCK-END

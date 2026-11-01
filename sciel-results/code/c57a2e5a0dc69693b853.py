import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _min_distance(centers):
    """Calculate the minimum distance between any two centers."""
    n = len(centers)
    min_dist = np.inf
    # Vectorized calculation is faster but O(N^2) memory. N=50 is small.
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    return min_dist

def _score(centers):
    """
    Score is the radius r.
    r = min(min_pairwise_dist / 2, min_dist_to_wall)
    """
    min_pair_dist = _min_distance(centers)
    
    # Distance to walls: center must be in [-0.5, 0.5]
    # dist_to_wall for point c is min(0.5 - c, c) for each coord, then min over coords?
    # Actually, the constraint is 1/2 - |c_ik| >= r.
    # So r <= 0.5 - max(|c_ik|) for each k? No.
    # r <= min_k (0.5 - |c_ik|) for each i.
    # So min_dist_to_wall = min_i ( min_k (0.5 - |c_ik|) )
    
    abs_centers = np.abs(centers)
    dist_to_walls = 0.5 - np.max(abs_centers, axis=1) # min distance from each center to its closest wall face
    min_wall_dist = np.min(dist_to_walls)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _objective(x_flat):
    """Objective function for minimization. We want to maximize score, so minimize -score."""
    centers = x_flat.reshape(-1, 3)
    # Penalize if points go outside the cube significantly to help optimizer
    # But Nelder-Mead doesn't use bounds. We can add a penalty.
    outside_penalty = 0.0
    if np.any(np.abs(centers) > 0.5):
        outside_penalty = 10.0 * np.sum(np.maximum(0, np.abs(centers) - 0.5))
    
    s = _score(centers)
    return -s + outside_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_centers = None
    
    # Strategy: Try a few initializations and optimize them.
    # Initialization 1: Grid-based
    # 4x4x4 grid has 64 points. We need 50.
    # Let's create a 4x4x4 grid and remove 14 points.
    # To maintain some symmetry, we can remove points from the corners/edges?
    # Or just take the first 50?
    # Better: Randomly select 50 from the 64 grid points, then perturb.
    
    rng = np.random.default_rng(42)
    
    # Create 4x4x4 grid
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 points randomly from the 64
    indices = rng.choice(64, size=50, replace=False)
    init_centers = grid_points[indices].copy()
    
    # Add small random perturbation to break symmetry
    init_centers += rng.normal(0, 0.01, init_centers.shape)
    
    # Clip to ensure they are inside (though optimizer should handle it, 
    # starting inside is safer)
    init_centers = np.clip(init_centers, -0.49, 0.49)
    
    # Run Local Optimization
    try:
        res = minimize(_objective, init_centers.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        opt_centers = res.x.reshape(-1, 3)
        # Ensure final result is strictly inside
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        # Re-clip to slightly inside to avoid floating point errors on boundary
        opt_centers = np.clip(opt_centers, -0.499999, 0.499999)
        
        s = _score(opt_centers)
        if s > best_score:
            best_score = s
            best_centers = opt_centers
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20.0:
        if best_centers is not None:
            return best_centers
        return init_centers[:N]

    # Try another initialization: Random start
    for _ in range(3): # Try 3 random restarts if time permits
        if time.time() > deadline - 15.0:
            break
        
        random_centers = rng.uniform(-0.4, 0.4, (N, 3))
        
        try:
            res = minimize(_objective, random_centers.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            opt_centers = res.x.reshape(-1, 3)
            opt_centers = np.clip(opt_centers, -0.499999, 0.499999)
            
            s = _score(opt_centers)
            if s > best_score:
                best_score = s
                best_centers = opt_centers
        except Exception:
            pass

    if best_centers is not None:
        return best_centers
    else:
        # Fallback to simple grid if everything failed
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initializations
    # 1. 5x5x4 Grid
    # x: 5 pts, y: 5 pts, z: 4 pts
    # Spacing 0.2
    x = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    z = np.array([-0.3, -0.1, 0.1, 0.3]) # Centered 4 pts?
    # Let's check z spacing.
    # If we use 0.2 spacing: -0.3, -0.1, 0.1, 0.3.
    # Wall dist: 0.2.
    # Inter dist: 0.2.
    # r = 0.1.
    # This is better than the seed's z=-0.4..0.2 (wall dist 0.1).
    # So this init already improves the score to 0.1 (same as seed max, but better centering).
    # Wait, seed had z min -0.4, wall dist 0.1.
    # This init has z min -0.3, wall dist 0.2.
    # So this init is strictly better or equal.
    
    grids = []
    # Grid 1: 5x5x4 centered
    x5 = np.linspace(-0.4, 0.4, 5)
    y5 = np.linspace(-0.4, 0.4, 5)
    z4 = np.linspace(-0.3, 0.3, 4)
    grid1 = np.array([(x, y, z) for z in z4 for y in y5 for x in x5])
    grids.append(grid1)
    
    # Grid 2: 4x5x5
    x4 = np.linspace(-0.3, 0.3, 4)
    y5b = np.linspace(-0.4, 0.4, 5)
    z5 = np.linspace(-0.4, 0.4, 5)
    grid2 = np.array([(x, y, z) for z in z5 for y in y5b for x in x4])
    grids.append(grid2)
    
    # Grid 3: 5x4x5
    x5c = np.linspace(-0.4, 0.4, 5)
    y4 = np.linspace(-0.3, 0.3, 4)
    z5b = np.linspace(-0.4, 0.4, 5)
    grid3 = np.array([(x, y, z) for z in z5b for y in y4 for x in x5c])
    grids.append(grid3)
    
    best_result = None
    best_score = 0.0
    
    def score(points):
        # Compute min radius
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        # min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        wall_dists = np.min(0.5 - np.abs(points), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)
    
    def objective(x_flat):
        points = x_flat.reshape(-1, 3)
        # Soft min approximation
        # d is the set of all distances (pair and wall)
        # We want to maximize min(d).
        # Equivalent to minimize -min(d).
        # Use softmax with beta
        beta = 50.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6) # Ignore self
        
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        # Combine all distances
        # Pair dists (upper triangle to avoid duplication, or all)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        # Soft min: -1/beta * ln(sum(exp(-beta * d)))
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        soft_min = -1.0/beta * np.log(np.sum(exp_terms)) + max_d
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min
    
    for grid in grids:
        if time.time() > deadline:
            break
            
        x0 = grid.flatten()
        
        # Bounds: [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            if time.time() > deadline:
                break
            points = res.x.reshape(-1, 3)
            s = score(points)
            if s > best_score:
                best_score = s
                best_result = points
        except Exception:
            pass
    
    if best_result is None:
        return grid1
        
    return best_result

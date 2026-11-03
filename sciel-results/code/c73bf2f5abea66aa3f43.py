import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill

def _score_from_points(pts):
    """Calculate the maximum radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Min distance to wall
    wall_dist = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    r = min(wall_dist, min_pair / 2.0)
    return r if r > 0 else 0.0

def _local_optimize(pts, time_budget):
    """
    Directly optimize the 400 coordinates to maximize the minimum distance.
    Uses a smooth repulsion objective: sum(1/d^4) + sum(1/w^4).
    """
    def objective(x):
        p = x.reshape(-1, 2)
        
        # Wall distances
        w_left = p[:, 0] + 0.5
        w_right = 0.5 - p[:, 0]
        w_bottom = p[:, 1] + 0.5
        w_top = 0.5 - p[:, 1]
        
        # Penalize if outside (though L-BFGS-B should keep them in bounds)
        # We use soft penalties in the objective too just in case
        w_pen = 0
        if np.any(w_left < 0): w_pen += np.sum(w_left[w_left < 0]**4)
        if np.any(w_right < 0): w_pen += np.sum(w_right[w_right < 0]**4)
        if np.any(w_bottom < 0): w_pen += np.sum(w_bottom[w_bottom < 0]**4)
        if np.any(w_top < 0): w_pen += np.sum(w_top[w_top < 0]**4)
        
        # Wall repulsion (push away from walls)
        # Avoid division by zero
        eps = 1e-8
        w_term = np.sum(1.0 / (w_left + eps)**4) + np.sum(1.0 / (w_right + eps)**4) + \
                 np.sum(1.0 / (w_bottom + eps)**4) + np.sum(1.0 / (w_top + eps)**4)
                 
        # Pairwise repulsion
        D = cdist(p, p)
        np.fill_diagonal(D, np.inf)
        # Only sum close pairs to save time? No, vectorized sum is fast enough for N=200.
        # But 1/d^4 can explode.
        pair_term = np.sum(1.0 / (D + eps)**4)
        
        return pair_term + w_term + w_pen

    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        # Use L-BFGS-B for speed and bound handling
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        return res.x.reshape(-1, 2)
    except Exception:
        return pts

def _generate_grid(rows, cols, jitter=0.0):
    """Generate a grid of points inside [-0.5, 0.5]^2."""
    if rows * cols < N:
        return None
    
    # Center the grid
    dx = 1.0 / cols
    dy = 1.0 / rows
    
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    
    grid = np.array([(x, y) for y in ys for x in xs])
    
    # If we have more than N, we need to select N.
    # Usually, we pick the ones closest to the center to maximize interior packing.
    if len(grid) > N:
        dists = np.sum(grid**2, axis=1)
        indices = np.argsort(dists)[:N]
        grid = grid[indices]
    
    # Apply jitter
    if jitter > 0:
        grid += np.random.normal(0, jitter, grid.shape)
        # Clip to ensure they are inside, with a small margin
        grid[:, 0] = np.clip(grid[:, 0], -0.49, 0.49)
        grid[:, 1] = np.clip(grid[:, 1], -0.49, 0.49)
        
    return grid

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Structural Change: Direct coordinate optimization from multiple grid starts
    # Grid configurations: (rows, cols)
    configs = [
        (14, 15), (15, 14),
        (13, 16), (16, 13),
        (12, 17), (17, 12),
        (10, 20), (20, 10),
        (15, 15), # 225 points
        (14, 14), # 196 points (too few, skip? No, 14*14=196 < 200, so this is invalid)
    ]
    
    # Filter valid configs
    valid_configs = [(r, c) for r, c in configs if r * c >= N]
    
    # Number of restarts per config
    # We have ~145s. Each local opt takes ~1-2s. 
    # Let's do 3 restarts per config initially, then increase if time permits.
    num_restarts_per_config = 3
    
    for rows, cols in valid_configs:
        if time.time() - t0 > _TIME_LIMIT - 20: # Leave time for final check
            break
            
        for k in range(num_restarts_per_config):
            if time.time() - t0 > _TIME_LIMIT - 10:
                break
                
            # Generate initial points
            # Jitter magnitude: small enough to keep structure, large enough to break symmetry
            jitter = 0.01 + 0.01 * k
            pts = _generate_grid(rows, cols, jitter=jitter)
            
            if pts is None or len(pts) != N:
                continue
            
            # Local optimization
            optimized_pts = _local_optimize(pts, time_budget=5.0)
            
            # Evaluate
            r = _score_from_points(optimized_pts)
            
            if r > best_r:
                best_r = r
                best_pts = optimized_pts
                # Print progress (optional, but good for debugging if allowed, but we shouldn't print in production usually)
                # print(f"Improved: {rows}x{cols} restart {k}, r={r}")
                
            # If we have time, do a second pass with smaller jitter on the best found so far
            if time.time() - t0 < _TIME_LIMIT - 15:
                 # Refine the current best with very small jitter
                 if best_pts is not None:
                     refined_pts = best_pts + np.random.normal(0, 0.001, best_pts.shape)
                     refined_pts[:, 0] = np.clip(refined_pts[:, 0], -0.49, 0.49)
                     refined_pts[:, 1] = np.clip(refined_pts[:, 1], -0.49, 0.49)
                     refined_pts = _local_optimize(refined_pts, time_budget=3.0)
                     r_refined = _score_from_points(refined_pts)
                     if r_refined > best_r:
                         best_r = r_refined
                         best_pts = refined_pts

    # Fallback if nothing worked (should not happen)
    if best_pts is None:
        pts = _generate_grid(14, 15, jitter=0.0)
        if pts is not None and len(pts) == N:
            best_pts = pts
        else:
            # Absolute fallback
            best_pts = np.zeros((N, 2))
            
    return best_pts

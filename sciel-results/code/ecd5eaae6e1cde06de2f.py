import numpy as np
import scipy.optimize
import time
from scipy.spatial.distance import cdist

N = 50

def _score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) != N:
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(np.abs(centers) > 0.5):
        return -np.inf
        
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Vectorized pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def generate_hex_init():
    # Estimate d for N=50
    # d ~ sqrt(2 / (N * sqrt(3)))
    d = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Add a small buffer to ensure we get at least N points
    d *= 0.95 
    
    row_height = d * np.sqrt(3.0) / 2.0
    n_rows = int(np.ceil(1.0 / row_height)) + 2
    n_cols = int(np.ceil(1.0 / d)) + 2
    
    centers = []
    for i in range(n_rows):
        y = -0.5 + i * row_height
        if i % 2 == 0:
            x_start = -0.5
        else:
            x_start = -0.5 + d/2.0
            
        for j in range(n_cols):
            x = x_start + j * d
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Clip to square
    centers = np.clip(centers, -0.5, 0.5)
    
    # If we have more than N points, remove the ones closest to the boundary
    if len(centers) > N:
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        # Sort by distance to boundary, smallest first
        sort_idx = np.argsort(dist_to_boundary)
        # Keep the last N points (largest distance to boundary)
        centers = centers[sort_idx[-N:]]
        
    # If we have fewer than N points (should not happen with buffer), pad with random
    if len(centers) < N:
        # This case is unlikely, but handle it
        extra = np.random.uniform(-0.4, 0.4, (N - len(centers), 2))
        centers = np.vstack([centers, extra])

    return centers

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Initial Hex Grid
    init_centers = generate_hex_init()
    x0 = init_centers.flatten()
    
    # 2. Objective Function
    def objective(x):
        c = x.reshape(-1, 2)
        # We don't clip here to keep the function smooth for the optimizer
        # but we rely on the boundary penalty
        dist_bound = np.min(0.5 - np.abs(c), axis=1)
        min_bound = np.min(dist_bound)
        
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_bound)
        return -r
        
    # 3. Optimization
    # Powell is good for this
    # Nelder-Mead is also an option
    # Let's try Powell first
    
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            method='Powell', 
            options={'maxiter': 500, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
        )
        best_centers = res.x.reshape(-1, 2)
        best_score = -res.fun
    except Exception:
        best_centers = init_centers
        best_score = _score(init_centers)
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Not enough time for more, return best
        best_centers = np.clip(best_centers, -0.5, 0.5)
        return best_centers
        
    # 4. Try a second optimization with Nelder-Mead on the result
    # Or a random restart?
    # Let's try Nelder-Mead on the Powell result to polish
    try:
        res2 = scipy.optimize.minimize(
            objective, 
            best_centers.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if -res2.fun > best_score:
            best_centers = res2.x.reshape(-1, 2)
            best_score = -res2.fun
    except Exception:
        pass
        
    # Final Clip
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

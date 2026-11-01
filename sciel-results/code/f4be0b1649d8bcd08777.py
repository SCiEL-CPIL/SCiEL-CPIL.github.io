import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # 1. Initial Guess: Hexagonal Grid
    # Generate a 6x6 hex grid
    r_init = 0.09
    dx = np.sqrt(3) * r_init
    dy = 1.5 * r_init
    
    points = []
    for i in range(6):
        for j in range(6):
            x = j * dx + (i % 2) * dx / 2
            y = i * dy
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    points -= np.mean(points, axis=0)
    
    # Scale to fit in [0.1, 0.9] roughly
    # Current extent
    min_p = np.min(points, axis=0)
    max_p = np.max(points, axis=0)
    width = max_p[0] - min_p[0]
    height = max_p[1] - min_p[1]
    scale = 0.8 / max(width, height)
    points *= scale
    
    # Center again after scaling
    points -= np.mean(points, axis=0)
    points += 0.5
    
    # We have 36 points. We need 32.
    # Remove 4 points. Which ones?
    # The ones farthest from the center of the square (0.5, 0.5) are likely the corners.
    # Removing corners might allow the remaining circles to expand towards the corners.
    dist_to_center = np.sum((points - 0.5)**2, axis=1)
    # Indices of the 4 farthest points
    farthest_idx = np.argsort(dist_to_center)[-4:]
    # Create mask
    mask = np.ones(36, dtype=bool)
    mask[farthest_idx] = False
    init_centers = points[mask]
    
    # Ensure strictly inside
    init_centers = np.clip(init_centers, 0.01, 0.99)
    
    # 2. Objective Function
    # Smooth min approximation: min(a,b) ~ (a+b - sqrt((a-b)^2 + eps^2)) / 2
    def smooth_min(a, b, eps=1e-4):
        return (a + b - np.sqrt((a - b)**2 + eps**2)) / 2.0

    def objective(flat_x):
        x = flat_x.reshape(N, 2)
        
        # Distances to boundaries
        dist_bound = np.minimum(np.minimum(x[:, 0], 1 - x[:, 0]), np.minimum(x[:, 1], 1 - x[:, 1]))
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Min distance to other circles / 2
        min_dist_other = np.min(dists, axis=1) / 2.0
        
        # Smooth min of boundary dist and neighbor dist
        radii = smooth_min(dist_bound, min_dist_other)
        
        # We want to maximize sum of radii
        return -np.sum(radii)
    
    x0 = init_centers.ravel()
    
    best_x = x0
    best_val = objective(x0)
    
    # 3. Optimization
    # Use Powell
    try:
        # maxiter: Powell uses "iterations" which are cycles of line searches.
        # For 64 vars, 100 iterations might be enough for a decent start.
        # Let's try 200.
        result = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'disp': False, 'xtol': 1e-5, 'ftol': 1e-5})
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
            
    except Exception:
        pass
    
    # If we have time, do a second pass with Nelder-Mead?
    if time.time() < deadline - 10:
        try:
            result2 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 500, 'disp': False, 'xatol': 1e-4, 'fatol': 1e-6})
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass

    # 4. Final Calculation
    final_centers = best_x.reshape(N, 2)
    final_centers = np.clip(final_centers, 0, 1)
    
    # Recalculate exact radii (no smoothing)
    dist_bound = np.minimum(np.minimum(final_centers[:, 0], 1 - final_centers[:, 0]), np.minimum(final_centers[:, 1], 1 - final_centers[:, 1]))
    
    diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dist_other = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(dist_bound, min_dist_other)
    radii = np.maximum(radii, 0)
    
    return np.column_stack([final_centers, radii])

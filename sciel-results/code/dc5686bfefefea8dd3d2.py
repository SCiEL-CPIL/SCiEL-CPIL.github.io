import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    best_c = None
    best_val = -np.inf
    
    # Generate diverse initializations
    inits = []
    # Grids
    grid_configs = [(6, 6), (5, 7), (7, 5), (4, 8), (8, 4), (8, 8)]
    for r, c in grid_configs:
        inits.append(create_grid_init(r, c, N))
        
    # Random
    for _ in range(10):
        inits.append(np.random.uniform(0.05, 0.95, size=N*2))
        
    # Perturbed Grids
    base_grid = create_grid_init(6, 6, N)
    for noise_scale in [0.05, 0.1, 0.2]:
        noise = np.random.normal(0, noise_scale, size=base_grid.shape)
        inits.append(np.clip(base_grid + noise, 0.01, 0.99))

    # Stage 1: Fast local search on all inits to find promising candidates
    candidates = []
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20: # Reserve time for stage 2
            break
            
        # Quick Nelder-Mead
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            val = -res.fun
            candidates.append((val, res.x))
        except:
            candidates.append((sum_radii(c_init), c_init))
            
    # Sort candidates by value
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Stage 2: Refine top K candidates with more powerful Powell
    K = 5
    for val, c_curr in candidates[:K]:
        if time.time() - start_time > timeout - 5:
            break
            
        # Powell is good for non-smooth but needs more iterations
        try:
            res = minimize(lambda c: -sum_radii(c), c_curr, method='Powell',
                          options={'maxiter': 500, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            if val > best_val:
                best_val = val
                best_c = c_curr.copy()
                
    # If no valid result, fall back to best candidate or grid
    if best_c is None:
        if candidates:
            best_val, best_c = candidates[0]
        else:
            best_c = create_grid_init(6, 6, N)
            best_val = sum_radii(best_c)
            
    # Final polish with Nelder-Mead on the best solution
    if time.time() - start_time < timeout - 2:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass

    # Convert to (N, 3) array
    p = best_c.reshape(N, 2)
    # Recalculate radii for the final centers
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result

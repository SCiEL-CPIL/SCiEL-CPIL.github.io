import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    """
    Compute the maximum radius for the given centers.
    Returns (radius, valid)
    """
    # Check bounds
    # centers shape (N, 2)
    # Square is [-0.5, 0.5]
    if not np.all(np.isfinite(centers)):
        return 0.0, False
    
    # Distance to walls
    # min distance from center to any wall
    # wall_x_left = -0.5, wall_x_right = 0.5
    # dist_x = min(center_x - (-0.5), 0.5 - center_x)
    # dist_y = min(center_y - (-0.5), 0.5 - center_y)
    # wall_dist = min(dist_x, dist_y)
    
    # Vectorized calculation
    # dist to left wall: centers[:, 0] + 0.5
    # dist to right wall: 0.5 - centers[:, 0]
    # dist to bottom wall: centers[:, 1] + 0.5
    # dist to top wall: 0.5 - centers[:, 1]
    
    d_x_left = centers[:, 0] + 0.5
    d_x_right = 0.5 - centers[:, 0]
    d_y_bottom = centers[:, 1] + 0.5
    d_y_top = 0.5 - centers[:, 1]
    
    wall_dists = np.minimum.reduce([d_x_left, d_x_right, d_y_bottom, d_y_top])
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0, False
        
    # Pairwise distances
    # Using broadcasting for (N, N) matrix
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    
    # To save memory and time, we can do it in chunks or just use the full matrix since N=50 is small
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    return r, True

def penalty_function(params, R):
    """
    Penalty function for radius R.
    params: flattened array of length 2*N
    """
    centers = params.reshape(N, 2)
    
    # Wall violations
    # centers should be in [-0.5+R, 0.5-R]
    # violation = max(0, |center| - (0.5 - R))
    # Actually, distance to wall < R means violation.
    # dist_to_wall = 0.5 - |center|
    # if dist_to_wall < R, violation = R - dist_to_wall
    
    # Vectorized wall penalty
    # d_x = 0.5 - np.abs(centers[:, 0])
    # d_y = 0.5 - np.abs(centers[:, 1])
    # wall_viol_x = np.maximum(0, R - d_x)
    # wall_viol_y = np.maximum(0, R - d_y)
    # wall_penalty = np.sum(wall_viol_x**2 + wall_viol_y**2)
    
    # Simpler: just push them in if they are too close to wall
    # Let's use the explicit distance formulation
    d_x = 0.5 - np.abs(centers[:, 0])
    d_y = 0.5 - np.abs(centers[:, 1])
    
    wall_viol_x = np.maximum(0, R - d_x)
    wall_viol_y = np.maximum(0, R - d_y)
    
    wall_penalty = np.sum(wall_viol_x**2 + wall_viol_y**2)
    
    # Pairwise violations
    # dists < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Only consider i < j to avoid double counting and self
    i, j = np.triu_indices(N, k=1)
    pair_dists = dists[i, j]
    
    pair_viol = np.maximum(0, 2*R - pair_dists)
    pair_penalty = np.sum(pair_viol**2)
    
    return wall_penalty + pair_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Keep 20s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Jitter for multiple restarts
    num_restarts = 5
    
    # We will try to increase R from a low value
    # Start with R slightly above grid radius
    grid_r = 0.0625
    R_low = grid_r
    R_high = 0.072
    
    # Bisection on R
    # For each R, we try to find a configuration that is feasible (penalty ~ 0)
    # We use the best found centers as initial guess for the next R
    
    current_centers = initial_centers.copy()
    
    # To make it faster, we can do a coarse search first
    # Or just iterate R up
    
    # Let's do a simple hill-climbing on R
    # Start with R = grid_r * 1.01
    R = grid_r * 1.01
    
    # Optimization settings
    options = {'maxiter': 200, 'ftol': 1e-8}
    
    while R < R_high:
        if time.time() > deadline - 10.0:
            break
            
        success = False
        best_pen = np.inf
        best_x = current_centers.flatten()
        
        # Try a few random perturbations of the current best as start points
        # Or just use current best
        starts = [current_centers.flatten()]
        
        # Add a few random perturbations if we have time
        if time.time() < deadline - 30.0:
            for _ in range(2):
                perturbed = current_centers + np.random.normal(0, 0.01, (N, 2))
                # Keep within bounds roughly
                perturbed = np.clip(perturbed, -0.45, 0.45)
                starts.append(perturbed.flatten())
                
        for x0 in starts:
            if time.time() > deadline - 10.0:
                break
                
            res = minimize(
                penalty_function, 
                x0, 
                args=(R,), 
                method='L-BFGS-B', 
                options=options
            )
            
            if res.fun < best_pen:
                best_pen = res.fun
                best_x = res.x
                
            # If penalty is very low, we might have found a feasible solution
            if best_pen < 1e-6:
                success = True
                break
                
        if success:
            # Update best centers
            current_centers = best_x.reshape(N, 2)
            # Calculate actual radius to update best
            r_actual, valid = compute_radius(current_centers)
            if valid and r_actual > best_radius:
                best_radius = r_actual
                best_centers = current_centers.copy()
                
            # Increase R
            R += 0.0005
        else:
            # If not feasible, we might be stuck or R is too high
            # Try a slightly smaller step or break
            # If we can't find a feasible solution for R, it's likely R is too high for this local optimum
            # We can try to restart from a different grid if we haven't succeeded in a while
            # For now, just break out of this local search and try a new restart
            break

    # If we haven't found anything better than grid, return grid
    if best_centers is None:
        return initial_centers
        
    return best_centers

# Run construct to test? No, just define it.
# The evaluator calls construct().

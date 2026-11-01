import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_radius(centers):
    # Calculate the minimum distance between any two centers
    # and the minimum distance from any center to the boundary.
    # centers: (N, 3) array
    if centers.shape != (N, 3):
        return 0.0
    
    # Pairwise distances
    # To avoid O(N^2) memory blowup for large N, we can do it in chunks, 
    # but N=100 is small enough for a (100, 100) matrix.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask out the diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    # Cube is [-0.5, 0.5]^3
    # Distance from center c to boundary in dimension k is 0.5 - |c_k|
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2, min_boundary_dist)

def soft_min_radius(centers, beta=50.0):
    # Approximate the minimum radius using a soft-min function.
    # We want to maximize r, so we minimize -r.
    # r = min(min_pair_dist/2, min_boundary_dist)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of pairwise distances
    # -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, we subtract the min
    min_d = np.min(dists)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (dists - min_d)))) + (-beta * min_d)
    soft_min_pair = -log_sum_exp / beta
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    min_b = np.min(boundary_dists)
    log_sum_exp_b = np.log(np.sum(np.exp(-beta * (boundary_dists - min_b)))) + (-beta * min_b)
    soft_min_boundary = -log_sum_exp_b / beta
    
    # The overall soft min is the min of the two soft mins
    # Approximate min(a, b) as -1/beta * log(exp(-beta*a) + exp(-beta*b))
    min_ab = min(soft_min_pair, soft_min_boundary)
    log_sum_exp_ab = np.log(np.exp(-beta * (soft_min_pair - min_ab)) + np.exp(-beta * (soft_min_boundary - min_ab))) + (-beta * min_ab)
    soft_min_total = -log_sum_exp_ab / beta
    
    return -soft_min_total / 2  # We want to minimize the negative radius

def objective(centers_flat, beta=50.0):
    centers = centers_flat.reshape(N, 3)
    return soft_min_radius(centers, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Random points
    for i in range(5):
        np.random.seed(i)
        inits.append(np.random.uniform(-0.5, 0.5, (N, 3)))
        
    # 2. Grid with perturbations
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    for i in range(3):
        np.random.seed(100 + i)
        perturbation = np.random.uniform(-0.05, 0.05, (N, 3))
        inits.append(grid + perturbation)
        
    # 3. Simple cubic grid (baseline)
    inits.append(grid)
    
    # Optimize each initial configuration
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B
            # Bounds: [-0.5, 0.5] for each coordinate
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            res = minimize(
                objective, 
                init.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                centers = res.x.reshape(N, 3)
                # Ensure centers are strictly inside the cube to avoid numerical issues
                # The optimizer should handle this with bounds, but let's clip just in case
                centers = np.clip(centers, -0.5, 0.5)
                
                r = compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers
                    
        except Exception as e:
            pass
            
    # If no optimization succeeded, return the grid
    if best_centers is None:
        return grid
        
    return best_centers

# EVOLVE-BLOCK-START
# (The code above is the new design)
# EVOLVE-BLOCK-END

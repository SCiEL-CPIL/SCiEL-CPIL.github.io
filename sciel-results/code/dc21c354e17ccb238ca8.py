"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin for 120s limit
    
    # 1. Initial Configuration
    # Use a dense grid and select N points, then add jitter to break symmetry
    m = 4 # 4x4x4 = 64 points, enough to pick 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle and take N
    np.random.seed(42)
    indices = np.random.choice(len(grid), N, replace=False)
    centers = grid[indices].copy()
    
    # Add small jitter to break grid symmetry
    centers += np.random.normal(0, 0.01, size=(N, 3))
    
    # Clip to ensure inside cube initially
    centers = np.clip(centers, -0.49, 0.49)
    
    def min_radius(c):
        # Compute pairwise distances
        # c is (N, 3)
        # diff is (N, N, 3)
        # dist is (N, N)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist)
        
        # Distance to walls
        # c is in [-0.5, 0.5]
        # dist to wall is min(0.5 - c, c + 0.5) for each coord
        # Actually, distance from center to face x=0.5 is 0.5 - x
        # distance from center to face x=-0.5 is x - (-0.5) = x + 0.5
        # So min(0.5 - x, 0.5 + x)
        wall_dists = np.min(np.stack([0.5 - c, 0.5 + c], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # 2. Simulated Annealing
    # Vectorized update
    current_radius = min_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    T = 0.05
    cooling_rate = 0.995
    step_size = 0.05
    
    while T > 0.001 and (time.time() - start_time) < deadline - 20:
        # Generate random moves
        moves = np.random.normal(0, step_size, size=(N, 3))
        new_centers = centers + moves
        new_centers = np.clip(new_centers, -0.49, 0.49)
        
        new_radius = min_radius(new_centers)
        
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Revert
            pass
            
        T *= cooling_rate
        
        # Adaptive step size
        if T < 0.01:
            step_size *= 0.95

    centers = best_centers
    current_radius = best_radius
    
    # 3. Active Set Polish
    # Identify active constraints
    # We want to maximize the minimum distance.
    # Let's define a function to compute the "active" set and optimize locally.
    
    def objective(c):
        # Negative of the minimum radius
        return -min_radius(c)
    
    # We will run a few iterations of identifying active pairs and polishing
    for iter in range(5):
        if (time.time() - start_time) > deadline - 10:
            break
            
        c = centers
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        r = min_dist / 2.0
        
        # Find pairs that are close to the minimum
        # Threshold slightly larger than min to include "active" constraints
        threshold = min_dist * 1.01
        active_pairs = np.argwhere(dist < threshold)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Also check walls
        wall_dists = np.min(np.stack([0.5 - c, 0.5 + c], axis=1), axis=1)
        min_wall = np.min(wall_dists)
        
        if r < min_wall:
            # Sphere-sphere constraint is active
            # Optimize all centers, but focus on active pairs?
            # L-BFGS-B on all centers is fine for N=50
            res = so.minimize(objective, c, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 500})
            if res.fun < objective(c):
                centers = res.x
                centers = np.clip(centers, -0.49, 0.49)
        else:
            # Wall constraint is active
            # Optimize centers to push away from walls
            res = so.minimize(objective, c, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 500})
            if res.fun < objective(c):
                centers = res.x
                centers = np.clip(centers, -0.49, 0.49)
                
        # Check improvement
        new_r = min_radius(centers)
        if new_r <= current_radius:
             # If no improvement, break
             break
        current_radius = new_r

    return centers

# EVOLVE-BLOCK-END

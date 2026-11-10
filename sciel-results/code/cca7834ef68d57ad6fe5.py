"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Distance to walls
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dists)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return r_wall
    
    # Use broadcasting for pairwise distances (might be memory heavy for large N, but N=100 is fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def construct():
    # Use a body-centered cubic (BCC) lattice arrangement or similar high-symmetry packing
    # For N=100, we can try to arrange them in a grid that's denser than simple cubic.
    
    # Strategy: Use a 5x5x4 grid (100 points) but optimize positions
    # Or use a more sophisticated approach: start from a good initial guess and refine
    
    # Let's try a body-centered cubic style arrangement
    # In a unit cube, BCC has points at corners and body center
    # For 100 points, we can use a 5x5x4 grid which is 100 points exactly
    
    # Initial guess: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Optimize to maximize the minimum radius
    # We'll use a simple iterative optimization: for each point, try to move it 
    # to increase the minimum distance to all other points and walls
    
    # Better approach: Use scipy's minimize to maximize the minimum radius
    # Since we want to maximize a non-smooth function (min of distances), we'll use
    # a heuristic: iteratively move each point in the direction that increases 
    # its distance to its nearest neighbor
    
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Iterative local optimization
    for iteration in range(500):
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    for dz in [-0.01, 0, 0.01]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        new_center = best_centers[i].copy()
                        new_center[0] += dx
                        new_center[1] += dy
                        new_center[2] += dz
                        
                        # Check if still in cube
                        if np.any(np.abs(new_center) > 0.5):
                            continue
                        
                        test_centers = best_centers.copy()
                        test_centers[i] = new_center
                        r = _compute_radius(test_centers)
                        
                        if r > best_r + 1e-10:
                            best_centers = test_centers
                            best_r = r
                            improved = True
        if not improved:
            break
    
    return best_centers
# EVOLVE-BLOCK-END

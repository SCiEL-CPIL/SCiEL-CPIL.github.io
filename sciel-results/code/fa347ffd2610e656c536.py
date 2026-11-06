"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # We want to maximize the minimum distance between any two sphere centers,
    # subject to all centers being inside [-0.5, 0.5]^3.
    # The radius is half the minimum pairwise distance, and also limited by
    # the distance to the walls: r <= 0.5 - max(|x|, |y|, |z|) for each center.
    
    # Strategy: Use a simulated annealing / gradient-free optimization approach
    # to maximize the minimum of:
    #   - all pairwise distances / 2
    #   - distances from each center to the nearest wall (which is 0.5 - max(|c_i_k|))
    
    # Since scipy.optimize.minimize with Nelder-Mead can be slow for 150 dims,
    # let's use a simpler approach: start from a good initial configuration and
    # do local search via random perturbations with acceptance based on improvement.
    
    # Better: use a two-stage approach
    # 1. Start with a reasonable initial guess (e.g., face-centered cubic or body-centered cubic lattice points)
    # 2. Optimize using a custom local search
    
    # Let's try starting from a BCC (body-centered cubic) arrangement or FCC
    # For 50 spheres, a simple approach: use a 3x4x5 grid with some jitter, then optimize.
    
    # Actually, let's use a more systematic approach:
    # Start with points on a grid, then use scipy.optimize.minimize with Nelder-Mead
    # to maximize the minimum distance (by minimizing the negative).
    
    # The objective: minimize -min( min_{i<j} ||c_i - c_j||/2, min_i (0.5 - max(|c_i|)) )
    # This is non-smooth, so Nelder-Mead is appropriate.
    
    # Initial guess: distribute 50 points roughly uniformly
    # Use a 4x4x4 grid = 64 points, take first 50, or use 3x4x5=60, take 50.
    # Let's use a 4x4x4 grid and take 50 points, centered.
    
    # Grid approach for initialization:
    # 4x4x4 = 64 positions in [-0.5, 0.5]^3
    # Spacing: 1/4 = 0.25, positions at -0.375, -0.125, 0.125, 0.375
    
    # But we need exactly 50. Let's use a 3x4x5 = 60 grid, take 50.
    # Or better: use a 5x5x4 = 100 grid, take 50 with some spacing.
    
    # Simpler: just use a 4x4x4 grid, take the first 50 points.
    x_vals = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)  # -0.375, -0.125, 0.125, 0.375
    points = []
    for z in x_vals:
        for y in x_vals:
            for x in x_vals:
                points.append([x, y, z])
    points = np.array(points[:50])
    
    # Now optimize
    def objective(x):
        # x is flattened array of shape (150,)
        c = x.reshape(N, 3)
        # Check if all points are inside the cube
        # If any point is outside, return a large penalty
        if np.any(np.abs(c) > 0.5):
            return 1e10
        # Compute all pairwise distances
        # For efficiency, compute min pairwise distance
        # Use broadcasting: c[:, np.newaxis, :] - c[np.newaxis, :, :]
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask the diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Distance to walls
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        min_wall_dist = np.min(wall_dists)
        # The radius is min(min_pair_dist / 2, min_wall_dist)
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return -r
    
    # Use Nelder-Mead
    # This might be slow for 150 dimensions. Let's try it with a time limit in mind.
    # Nelder-Mead can be slow. Let's try a different approach: 
    # Use a simple local search with random restarts.
    
    # Alternative: use scipy.optimize.differential_evolution which is more robust
    # but also slow for 150 dims.
    
    # Let's try a hybrid: use the grid as initial, then do a few iterations of
    # "push apart" local optimization.
    
    # Actually, let's just use Nelder-Mead with a reasonable maxiter.
    # 150 dimensions is a lot. Let's try it.
    
    x0 = points.flatten()
    
    # Run optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    c = result.x.reshape(N, 3)
    
    # Ensure all points are strictly inside the cube
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return c
# EVOLVE-BLOCK-END

"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    """
    Uses Differential Evolution to maximize the minimum radius for 50 circles
    inside a unit square [-0.5, 0.5]^2.
    
    The objective function to maximize is:
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    
    We minimize -r instead.
    """
    def objective(x):
        # x is a flat array of length 2*N
        pts = x.reshape(N, 2)
        
        # Check if any point is outside the square (should be handled by bounds, but just in case)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1000.0  # Large penalty
        
        # Wall distances
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        min_wall_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        # Pairwise distances
        # Use efficient computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)  # Set diagonal to large value
        min_pair_dist = np.min(dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        return -r  # We want to maximize r, so minimize -r

    # Initial guess: Jittered grid
    rng = np.random.RandomState(42)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(float)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    # Flatten for differential_evolution
    x0 = points.flatten()
    
    # Bounds for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Run differential evolution
    # maxiter is the number of generations
    # popsize is the population size (multiplier)
    # tol is the tolerance for termination
    # seed for reproducibility
    # polish: use L-BFGS-B to polish the result
    
    result = differential_evolution(
        objective,
        bounds,
        seed=123,
        maxiter=1000,
        popsize=15,
        tol=1e-10,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=True
    )
    
    # Extract the best solution
    best_points = result.x.reshape(N, 2)
    
    # Ensure the points are strictly within the square (clip to be safe)
    best_points = np.clip(best_points, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_points
# EVOLVE-BLOCK-END

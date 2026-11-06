"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in [-0.5, 0.5]^3
    maximizing the minimum radius. We use a multi-start local optimization
    approach: start from a reasonable grid-like configuration and then
    iteratively push spheres apart to maximize the minimum pairwise distance
    and minimum distance to boundary.
    
    The radius is r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|)).
    We maximize this by maximizing the minimum of all these constraints.
    """
    n = N
    # Use a multi-start strategy with different initial configurations
    
    best_centers = None
    best_r = 0.0
    
    def compute_r(centers):
        """Compute the radius given centers."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        # For each center, distance to each of the 6 faces
        # Face x=0.5: 0.5 - x, face x=-0.5: 0.5 + x, etc.
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        boundary_dists = np.min(np.stack([
            0.5 - np.abs(x),
            0.5 - np.abs(y),
            0.5 - np.abs(z)
        ], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def objective(params):
        """Objective to minimize: negative of the radius."""
        centers = params.reshape(n, 3)
        r = compute_r(centers)
        return -r
    
    def make_grid_initial(m, n):
        """Create a grid-based initial configuration."""
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # If we have more than n, take first n; if fewer, add random
        if len(grid) >= n:
            return grid[:n].copy()
        else:
            # Add random points
            extra = np.random.uniform(-0.4, 0.4, size=(n - len(grid), 3))
            return np.vstack([grid, extra])
    
    # Try multiple initial configurations and local optimizations
    rng = np.random.default_rng(42)
    
    # Strategy 1: Grid-based starts with different grid sizes
    for m in [4, 5, 6]:
        centers = make_grid_initial(m, n)
        # Add small random perturbation
        centers += rng.uniform(-0.01, 0.01, size=centers.shape)
        
        # Local optimization using L-BFGS-B
        result = minimize(objective, centers.flatten(), method='L-BFGS-B',
                         bounds=[(-0.5, 0.5)] * (n * 3),
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success or result.fun < -best_r:
            r = -result.fun
            if r > best_r:
                best_r = r
                best_centers = result.x.reshape(n, 3)
    
    # Strategy 2: Random starts
    for trial in range(5):
        centers = rng.uniform(-0.45, 0.45, size=(n, 3))
        
        result = minimize(objective, centers.flatten(), method='L-BFGS-B',
                         bounds=[(-0.5, 0.5)] * (n * 3),
                         options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.success or result.fun < -best_r:
            r = -result.fun
            if r > best_r:
                best_r = r
                best_centers = result.x.reshape(n, 3)
    
    # If no optimization succeeded, fall back to grid
    if best_centers is None:
        m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    # Final cleanup: ensure centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_r(centers):
    """Compute the radius: min of half-min-pair-distance and min-boundary-distance."""
    # Pairwise distances
    n = len(centers)
    # Use broadcasting for efficiency but memory-conscious
    # For n=100, 100x100 is fine
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distance
    # centers in [-0.5, 0.5]^3
    # distance to boundary = 0.5 - max(|x|, |y|, |z|) for each point
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    boundary_dists = 0.5 - max_abs
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair, min_boundary)

def _objective(centers_flat):
    """Negative radius objective for minimization."""
    centers = centers_flat.reshape(-1, 3)
    r = _compute_r(centers)
    return -r

def _initial_grid(n):
    """Create an initial grid of n points in the unit cube [-0.5, 0.5]^3."""
    # Use a simple grid
    m = int(np.ceil(n ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_centers = _initial_grid(N)
    best_r = _compute_r(best_centers)
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 0
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial points
        if restart == 0:
            centers = best_centers.copy()
        else:
            # Random initialization with some structure
            # Start from grid and add noise
            base = _initial_grid(N)
            noise_scale = 0.05 * (1 - restart / max_restarts)
            centers = base + rng.randn(N, 3) * noise_scale
            # Clip to stay inside cube with margin
            centers = np.clip(centers, -0.45, 0.45)
        
        # Local optimization using L-BFGS-B
        try:
            centers_flat = centers.flatten()
            
            # Bounds: each coordinate in [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            result = minimize(
                _objective,
                centers_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_r:
                new_centers = result.x.reshape(-1, 3)
                new_r = _compute_r(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    print(f"Restart {restart}: improved to r={best_r:.10f}")
        except Exception as e:
            pass
        
        # If we've hit a good radius, try fewer more targeted restarts
        if best_r > 0.11:
            max_restarts = max(10, max_restarts - 2)
    
    # Final polish: try a few more optimization passes from best
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            try:
                centers_flat = best_centers.flatten()
                bounds = [(-0.5, 0.5)] * (N * 3)
                result = minimize(
                    _objective,
                    centers_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                new_centers = result.x.reshape(-1, 3)
                new_r = _compute_r(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
            except:
                break
    
    return best_centers
# EVOLVE-BLOCK-END

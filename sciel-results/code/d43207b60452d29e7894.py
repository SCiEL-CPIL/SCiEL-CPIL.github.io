"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Start from a grid layout, then refine with pairwise repulsion.
    # The grid gives a good initial guess; refinement increases the minimum distance.
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Define objective: maximize the minimum distance (pairwise and to boundary)
    # We minimize the negative of the minimum distance, but that's non-smooth.
    # Instead, we use a smooth approximation: minimize sum of inverse squared distances
    # with a large weight on the closest pair, or use a soft-min approach.
    
    # Better: use a potential energy approach where we push apart close circles.
    # We want to maximize r = min(min_{i<j} d_ij/2, min_i dist_to_boundary_i)
    
    # Let's define a smooth objective that penalizes small distances.
    # Use: -log(sum(exp(-d_ij / sigma))) for pairwise, and similarly for boundary.
    # This is a soft-min that approximates the minimum distance.
    
    def objective(x):
        x = x.reshape(N, 2)
        # Clip to stay in bounds (soft constraint)
        x = np.clip(x, -0.5, 0.5)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        boundary = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        
        # Soft-min approximation with sigma
        sigma = 0.001
        # For pairwise: use the smallest few distances
        flat_dist = dist.flatten()
        flat_dist = flat_dist[~np.isinf(flat_dist)]
        if len(flat_dist) == 0:
            return 0.0
        
        # Combine pairwise and boundary distances
        all_dists = np.concatenate([flat_dist, boundary])
        
        # Soft min: -log(sum(exp(-d/sigma))) * sigma
        # This is approximately the minimum for small sigma
        max_d = np.max(all_dists)
        exp_terms = np.exp(-(all_dists - max_d) / sigma)
        soft_min = max_d - sigma * np.log(np.sum(exp_terms))
        
        return -soft_min
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(objective, x0.flatten(), method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    x_opt = result.x.reshape(N, 2)
    
    # Ensure we stay strictly inside the square
    x_opt = np.clip(x_opt, -0.4999, 0.4999)
    
    return x_opt
# EVOLVE-BLOCK-END

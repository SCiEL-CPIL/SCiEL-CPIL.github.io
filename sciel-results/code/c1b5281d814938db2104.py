"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization.
    
    Scores candidates by the hypervolume improvement they would add to the
    set of non-dominated (mu, sigma) points. This explicitly optimizes for
    a diverse set of trade-offs rather than a single scalarized compromise.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Normalize mu and sigma to [0, 1] range for hypervolume computation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    
    if mu_range < 1e-12:
        mu_norm = np.zeros(n)
    else:
        mu_norm = (mu - mu_min) / mu_range
    
    if sigma_range < 1e-12:
        sigma_norm = np.zeros(n)
    else:
        sigma_norm = (sigma - sigma_min) / sigma_range
    
    # For minimization, we want low mu and low sigma to be "good"
    # In hypervolume, we typically maximize, so we invert: high score = low mu, low sigma
    # Let's work with (1 - mu_norm, 1 - sigma_norm) where higher is better
    
    x_coords = 1.0 - mu_norm
    y_coords = 1.0 - sigma_norm
    
    # Compute hypervolume contribution for each point
    # A point dominates another if it is better in at least one dimension and not worse in any
    # For 2D, we can compute the hypervolume improvement efficiently
    
    # Sort by x coordinate (descending, since higher x is better)
    order = np.argsort(-x_coords)
    x_sorted = x_coords[order]
    y_sorted = y_coords[order]
    
    # Compute hypervolume contribution for each point
    # In 2D, the hypervolume contribution of point i is:
    # (x_i - x_{i+1}) * y_i, where x_{i+1} is the next point's x coordinate
    # But we need to account for the reference point (0, 0)
    
    # The hypervolume is the area under the Pareto front
    # For each point, its contribution is the rectangle it adds to the hypervolume
    
    # Compute the Pareto front contributions
    # A point is on the Pareto front if no other point dominates it
    # In 2D with both objectives to maximize, point i dominates point j if
    # x_i >= x_j and y_i >= y_j and (x_i > x_j or y_i > y_j)
    
    # We'll compute the hypervolume contribution for each point
    # by finding the area it contributes to the total hypervolume
    
    # Sort by x descending
    # For each point in sorted order, track the maximum y seen so far
    # The hypervolume contribution is (x_i - x_{i+1}) * y_i if y_i > max_y_so_far
    
    hv_contrib = np.zeros(n)
    max_y = 0.0
    
    # Process in order of decreasing x
    for idx in range(len(order)):
        i = order[idx]
        x_val = x_sorted[idx]
        y_val = y_sorted[idx]
        
        if y_val > max_y:
            # This point contributes to the hypervolume
            # Its contribution is the area from x_val to the next point's x
            if idx + 1 < len(order):
                next_x = x_sorted[idx + 1]
            else:
                next_x = 0.0
            
            # The contribution is (x_val - next_x) * y_val
            # But we need to be careful: the hypervolume is the integral
            hv_contrib[i] = (x_val - next_x) * y_val
            max_y = y_val
    
    # Scale contributions to get a reasonable score
    # The total hypervolume is the sum of all contributions
    total_hv = np.sum(hv_contrib)
    
    if total_hv > 1e-12:
        scores = hv_contrib / total_hv
    else:
        # Fallback to EI if hypervolume is degenerate
        eps = 1e-12
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

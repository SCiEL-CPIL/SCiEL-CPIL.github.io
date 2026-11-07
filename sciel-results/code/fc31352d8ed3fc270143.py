"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Diversity-Weighted Expected Improvement (DW-EI) acquisition function.
    
    Combines standard Expected Improvement with a penalty based on local
    density of candidates, encouraging exploration of under-explored regions.
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle potential NaN/Inf from numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Density-aware penalty:
    # Estimate local density by looking at how many other candidates have
    # similar mu values. Candidates in "crowded" regions (many similar mu)
    # get penalized to encourage diversity.
    
    # Use a simple approach: bin candidates by mu value and count
    # The idea is that if many candidates have similar predicted means,
    # they are all exploring the same region, so we should discount them.
    
    n = len(mu)
    if n < 2:
        return ei
    
    # Create a density estimate using a kernel approach
    # For efficiency, use a simplified version:
    # Compute pairwise similarities to estimate local density
    # But with 2000 points, full pairwise is 4M ops which is fine
    
    # Use a fast approximation: sort by mu and use local neighborhood
    order = np.argsort(mu)
    sorted_mu = mu[order]
    
    # For each point, estimate density based on how many neighbors are close
    # Use a fixed bandwidth based on the spread of mu values
    mu_range = np.ptp(sorted_mu)
    if mu_range < 1e-10:
        density = np.ones(n)
    else:
        # Bandwidth: fraction of the total range
        bandwidth = 0.05 * mu_range
        
        # For each point, count how many others are within bandwidth
        # Use searchsorted for efficiency
        density = np.zeros(n)
        for i in range(n):
            lo = np.searchsorted(sorted_mu, sorted_mu[i] - bandwidth)
            hi = np.searchsorted(sorted_mu, sorted_mu[i] + bandwidth, side='right')
            density[order[i]] = hi - lo
        
        # Normalize density to [0, 1]
        density = density / (density.max() + 1e-9)
    
    # Weight EI by a diversity factor
    # High density (crowded region) -> lower score
    # Low density (sparse region) -> higher score
    # Use a moderate penalty to balance exploitation and exploration
    diversity_weight = 1.0 - 0.3 * density
    
    scores = ei * diversity_weight
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

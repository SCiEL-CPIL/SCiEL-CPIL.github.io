"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    n = len(mu)
    # UCB for minimization: maximize -mu + beta * sigma
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Structural change: Quantile binning of mu to enforce diversity
    # This prevents clustering by ensuring the selected point is the best
    # in its own quantile of the posterior mean distribution.
    
    # Sort indices by mu
    sorted_indices = np.argsort(mu)
    
    # Define number of bins
    n_bins = 20
    bin_size = n // n_bins
    
    scores = np.full(n, -np.inf)
    
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else n
        bin_indices = sorted_indices[start:end]
        
        # Find the index with the highest UCB score in this bin
        bin_scores = ucb[bin_indices]
        best_local_pos = np.argmax(bin_scores)
        best_global_idx = bin_indices[best_local_pos]
        
        # Assign the UCB score to this representative
        scores[best_global_idx] = ucb[best_global_idx]
        
    # Ensure all scores are finite
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
# EVOLVE-BLOCK-END

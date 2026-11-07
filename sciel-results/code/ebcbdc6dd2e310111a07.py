"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile-Bin UCB acquisition:
    1. Compute UCB scores.
    2. Partition candidates into K quantile bins based on posterior mean (mu).
    3. For each bin, only the candidate with the highest UCB score is "active" (high score).
    4. All other candidates in the bin are penalized heavily.
    5. This ensures the global argmax is one of the K bin winners, promoting diversity.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    n = len(mu)
    sigma = np.maximum(sigma, 1e-9)

    # Use a standard UCB with beta=1.0. 
    # Beale function is smooth, so moderate exploration is good.
    beta = 1.0
    ucb = mu - beta * sigma

    # Number of bins. 10 bins for 2000 points means 200 points per bin.
    # This ensures the "winners" are well-separated in mu-space.
    K = 10
    
    # Sort indices by mu to create quantile bins
    sorted_indices = np.argsort(mu)
    
    # Initialize scores with the UCB values
    scores = ucb.copy()
    
    # Determine bin boundaries
    # We want K roughly equal-sized bins
    bin_sizes = np.full(K, n // K)
    remainder = n % K
    # Distribute remainder to the first 'remainder' bins
    for i in range(remainder):
        bin_sizes[i] += 1
        
    start_idx = 0
    for k in range(K):
        end_idx = start_idx + bin_sizes[k]
        bin_indices = sorted_indices[start_idx:end_idx]
        
        if len(bin_indices) == 0:
            start_idx = end_idx
            continue
            
        # Find the index in the original array that has the max UCB in this bin
        # bin_indices are indices into the original arrays
        ucb_in_bin = ucb[bin_indices]
        local_max_pos = np.argmax(ucb_in_bin)
        winner_global_idx = bin_indices[local_max_pos]
        winner_ucb = ucb[winner_global_idx]
        
        # Penalize all non-winners in this bin
        # We want the winner to be the only one with a high score in this bin.
        # Set non-winners to a very low value so they don't compete globally.
        # We subtract a large constant from all points in the bin, then add it back for the winner.
        # Or simply: scores[bin_indices] = ucb[bin_indices] - 1000.0
        # scores[winner_global_idx] = ucb[winner_global_idx]
        
        # To keep the scale somewhat comparable, we can just subtract a penalty.
        # If we subtract 1000, and UCB values are in range [-10, 10], the winners will dominate.
        # This effectively creates K "super-candidates".
        
        mask = np.zeros(n, dtype=bool)
        mask[bin_indices] = True
        scores[mask] = ucb[bin_indices] - 1000.0
        
        # Restore the winner's score
        scores[winner_global_idx] = ucb[winner_global_idx]
        
        start_idx = end_idx

    # Ensure finite
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)

    # Deadline guard
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item using a spectral/variance-minimization approach.
    The goal is to minimize the variance of remaining capacities across all open bins,
    encouraging a homogeneous set of gaps that are statistically more likely to be
    filled by future items.
    
    For each valid bin, we compute the variance of the resulting remaining capacities
    after placing the item in that bin. We score each bin by the negative of this
    variance (lower variance = better = higher score).
    
    Higher score is better.
    """
    n_bins = len(bins)
    # Initialize all scores to a very low value (invalid for selection)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    # If no valid bins, return the low scores
    if not np.any(valid_mask):
        return scores
        
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = bins[valid_mask]
    
    # Compute the mean of current remaining capacities (for baseline)
    current_mean = np.mean(bins)
    
    # For each valid bin, compute the variance after placing the item there
    # We use an efficient computation: variance = E[X^2] - (E[X])^2
    # After placing item in bin i, the new remaining capacity for bin i is bins[i] - item
    # All other bins remain unchanged.
    
    # Precompute sum of squares and count for efficient variance calculation
    n = n_bins
    sum_sq = np.sum(bins ** 2)
    total_sum = np.sum(bins)
    
    # For each valid bin i, if we place item there:
    # new_sum = total_sum - item
    # new_sum_sq = sum_sq - bins[i]**2 + (bins[i] - item)**2
    # new_mean = new_sum / n
    # new_var = new_sum_sq / n - new_mean**2
    
    new_sum = total_sum - item
    new_mean = new_sum / n
    
    # For each valid bin, compute new_sum_sq efficiently
    # new_sum_sq_i = sum_sq - bins[i]**2 + (bins[i] - item)**2
    #             = sum_sq - bins[i]**2 + bins[i]**2 - 2*bins[i]*item + item**2
    #             = sum_sq - 2*bins[i]*item + item**2
    
    valid_new_sum_sq = sum_sq - 2.0 * valid_remaining * item + item ** 2
    valid_new_var = valid_new_sum_sq / n - new_mean ** 2
    
    # Score: negative variance (lower variance = higher score)
    scores[valid_mask] = -valid_new_var
    
    return scores

import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    
    Uses a variance-minimization heuristic: score each candidate bin by
    how well placing the item there maintains a homogeneous distribution
    of remaining capacities across all open bins. This reduces the entropy
    of the gap distribution, making gaps statistically more likely to be
    filled by future items.
    """
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    # Compute current mean and variance of all bin remainders
    all_remainders = bins.astype(np.float64)
    mean_all = np.mean(all_remainders)
    
    # For each valid bin, compute what the new variance would be
    # after placing the item in that bin
    # New remainder for bin i: bins[i] - item
    # Other bins unchanged
    
    # Vectorized computation of new variance for each candidate
    # Var_new = (1/n) * sum(x_j^2) - mean_new^2
    # where x_i = bins[i] - item, x_j = bins[j] for j != i
    
    sum_sq_all = np.sum(all_remainders ** 2)
    sum_all = np.sum(all_remainders)
    
    # For candidate i:
    # new_sum = sum_all - bins[i] + (bins[i] - item) = sum_all - item
    # new_sum_sq = sum_sq_all - bins[i]^2 + (bins[i] - item)^2
    # new_mean = (sum_all - item) / n_bins
    
    new_sum = sum_all - item
    new_mean = new_sum / n_bins
    
    # new_sum_sq for each valid candidate
    valid_remainders = all_remainders[valid_indices]
    new_sum_sq = sum_sq_all - valid_remainders ** 2 + (valid_remainders - item) ** 2
    
    # New variance for each candidate
    new_var = new_sum_sq / n_bins - new_mean ** 2
    
    # We want to MINIMIZE variance, so score = -variance
    # But we need to be careful: the variance is in absolute terms
    # Let's also add a best-fit tiebreaker for numerical stability
    
    # Primary score: negative variance (lower variance is better)
    variance_score = -new_var
    
    # Secondary: best-fit (minimize remaining after placement)
    remaining_after = valid_remainders - item
    best_fit_score = -remaining_after / 100.0  # Scale to be comparable
    
    # Blend: primarily variance, with small best-fit bias
    # The variance score should dominate as it captures the global structure
    blend = 0.95
    combined = blend * variance_score + (1.0 - blend) * best_fit_score
    
    scores[valid_indices] = combined
    
    return scores

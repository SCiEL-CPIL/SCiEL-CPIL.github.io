"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a probabilistic risk model
    based on variance reduction of the remaining capacity distribution.
    
    The approach models bin selection as minimizing the variance of remaining
    capacities across all open bins after placement. Bins that bring the
    remaining capacity distribution closer to a tighter, more uniform state
    reduce fragmentation and improve asymptotic packing density.
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    n = len(bins)
    if n == 0:
        return np.array([])
    
    remaining_after = bins - item
    valid_mask = remaining_after >= 0
    
    # If no valid bins, return all negative infinity
    if not np.any(valid_mask):
        return np.full(n, -np.inf)
    
    # Compute current mean of remaining capacities
    current_mean = np.mean(bins)
    
    # New mean after placing item in one of the valid bins
    # Only one bin changes, so new mean = (sum(bins) - item) / n
    new_mean = current_mean - item / n
    
    # For each bin, compute the change in total squared deviation (n * variance)
    # Old contribution: (bins[i] - current_mean)^2
    # New contribution for bin i if item placed there: (remaining_after[i] - new_mean)^2
    # Other bins unchanged, but their deviation from new_mean changes
    # 
    # Total old sum of squares: sum((bins - current_mean)^2)
    # Total new sum of squares if placed in bin i:
    #   (remaining_after[i] - new_mean)^2 + sum_{j != i} (bins[j] - new_mean)^2
    # 
    # We want to minimize the new total sum of squares, i.e., maximize the reduction
    # reduction_i = old_total - new_total_i
    # 
    # old_total = sum((bins - current_mean)^2)
    # new_total_i = (remaining_after[i] - new_mean)^2 + sum_{j != i} (bins[j] - new_mean)^2
    # 
    # Note: sum_{j != i} (bins[j] - new_mean)^2 = sum_{j} (bins[j] - new_mean)^2 - (bins[i] - new_mean)^2
    # 
    # So new_total_i = (remaining_after[i] - new_mean)^2 + sum_{j} (bins[j] - new_mean)^2 - (bins[i] - new_mean)^2
    # 
    # The term sum_{j} (bins[j] - new_mean)^2 is constant for all i, so it cancels in comparison
    # reduction_i = old_total - (remaining_after[i] - new_mean)^2 + (bins[i] - new_mean)^2
    # 
    # Let's compute this efficiently
    
    old_total = np.sum((bins - current_mean) ** 2)
    
    # For valid bins, compute the score as variance reduction
    # score_i = old_total - (remaining_after[i] - new_mean)^2 + (bins[i] - new_mean)^2
    
    # Compute for all bins (we'll mask out invalid ones)
    new_dev_sq = (remaining_after - new_mean) ** 2
    old_dev_at_new_mean = (bins - new_mean) ** 2
    
    # Score for each bin (valid or not, we'll mask)
    scores = old_total - new_dev_sq + old_dev_at_new_mean
    
    # Mask out invalid bins
    scores = np.where(valid_mask, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

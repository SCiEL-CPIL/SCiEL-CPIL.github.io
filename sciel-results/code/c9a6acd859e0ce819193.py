import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    Uses a spectral clustering / variance-minimization strategy:
    Score each candidate bin by how much placing the item there reduces the
    variance of remaining capacities across all open bins. This encourages
    a homogeneous set of gaps that are statistically more likely to be filled
    by future items, reducing fragmentation.
    """
    n_bins = len(bins)
    # Initialize all scores to a very low value (invalid for selection)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    # If no valid bins, return the low scores
    if not np.any(valid_mask):
        return scores
        
    # Extract remaining capacities of valid bins
    valid_remaining = bins[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if n_bins <= 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Compute current mean and variance of all bin remainders
    current_mean = np.mean(bins)
    current_var = np.var(bins)
    
    # For each valid bin i, after placing item:
    #   new_remaining[i] = bins[i] - item
    #   new_mean = (sum(bins) - item) / n_bins
    #   new_var = (1/n_bins) * sum((bins[j] - new_mean)^2 for all j)
    #
    # We want to MINIMIZE new_var, so score = -new_var (higher is better)
    #
    # new_var = (1/n) * [ sum_{j != i} (bins[j] - new_mean)^2 + (bins[i]-item - new_mean)^2 ]
    #
    # Let S = sum(bins), M = S/n (current mean)
    # new_mean = (S - item) / n = M - item/n
    #
    # Expand: sum_{j != i} (bins[j] - new_mean)^2 
    #   = sum_{j != i} [(bins[j] - M) + (M - new_mean)]^2
    #   = sum_{j != i} [(bins[j] - M) + item/n]^2
    #   = sum_{j != i} [(bins[j] - M)^2 + 2*(item/n)*(bins[j]-M) + (item/n)^2]
    #   = [sum_{j != i} (bins[j]-M)^2] + 2*(item/n)*[sum_{j != i}(bins[j]-M)] + (n-1)*(item/n)^2
    #
    # sum_{j != i} (bins[j]-M)^2 = n*current_var - (bins[i]-M)^2
    # sum_{j != i} (bins[j]-M) = 0 - (bins[i]-M) = -(bins[i]-M)
    #
    # And: (bins[i]-item - new_mean)^2 = (bins[i]-item - M + item/n)^2
    #   = (bins[i] - M - item*(1 - 1/n))^2
    #   = (bins[i] - M - item*(n-1)/n)^2
    #
    # So new_var = (1/n) * [
    #     n*current_var - (bins[i]-M)^2
    #     - 2*(item/n)*(bins[i]-M)
    #     + (n-1)*(item/n)^2
    #     + (bins[i]-M - item*(n-1)/n)^2
    #   ]
    #
    # Since we only need to rank valid bins, we can compute this per-bin.
    # The terms that don't depend on i can be dropped for ranking purposes,
    # but let's keep the full expression for clarity.
    
    # For efficiency, compute per-bin terms
    diff_from_mean = bins - current_mean  # shape (n_bins,)
    
    # For each valid bin i:
    # term1 = n*current_var - diff_from_mean[i]^2
    # term2 = -2*(item/n)*diff_from_mean[i]
    # term3 = (n-1)*(item/n)^2  (constant, doesn't affect ranking)
    # term4 = (diff_from_mean[i] - item*(n-1)/n)^2
    
    # score_i = -(term1 + term2 + term3 + term4) / n
    # Since term3/n is constant across all i, we can drop it for ranking.
    # score_i ∝ -(term1 + term2 + term4)
    
    # Compute for all bins (we'll mask invalid ones)
    term1 = n_bins * current_var - diff_from_mean ** 2
    term2 = -2.0 * (item / n_bins) * diff_from_mean
    shift = item * (n_bins - 1) / n_bins
    term4 = (diff_from_mean - shift) ** 2
    
    # Negative variance contribution (higher is better)
    # Drop the constant term3/n since it doesn't affect relative ranking
    raw_scores = -(term1 + term2 + term4)
    
    # Assign scores to valid bins
    scores[valid_mask] = raw_scores[valid_mask]
    
    return scores

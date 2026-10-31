import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full(len(bins), -np.inf)
    
    n = len(bins)
    total = np.sum(bins)
    sum_sq = np.sum(bins ** 2)
    
    new_total = total - item
    new_mean_sq = (new_total ** 2) / n
    
    # For each valid bin i, new variance = sum_sq - 2*bins[i]*item + item^2 - new_mean_sq
    # The term that varies with i is -2*bins[i]*item
    # So new_var[i] = (sum_sq + item^2 - new_mean_sq) - 2*bins[i]*item
    
    base = sum_sq + item ** 2 - new_mean_sq
    valid_indices = np.where(valid_mask)[0]
    valid_bins = bins[valid_indices]
    
    new_variances = base - 2 * valid_bins * item
    
    # Score: negative variance (lower variance is better)
    scores_valid = -new_variances
    
    # Add a small tie-breaker: prefer bins that leave less remaining space (best-fit)
    # This helps when variances are very close
    max_rem = np.max(remaining[valid_mask]) if np.any(valid_mask) else 1.0
    if max_rem > 0:
        tie_breaker = -0.001 * (remaining[valid_mask] / max_rem)
    else:
        tie_breaker = np.zeros(len(valid_indices))
    
    scores_valid = scores_valid + tie_breaker
    
    scores = np.full(n, -np.inf)
    scores[valid_mask] = scores_valid
    
    return scores

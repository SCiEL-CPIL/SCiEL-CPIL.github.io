"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Strategy: Spectral clustering / variance minimization approach.
    # Model the packing process as minimizing the variance of remaining capacities
    # across all open bins. Use a smooth surrogate (quadratic deviation from mean)
    # to approximate the effect of placing the item in each candidate bin.
    # This encourages homogeneous gap distributions that are more likely to be
    # filled by future items, reducing fragmentation.
    
    # Compute global statistics of remaining capacities across all bins
    mean_remaining = np.mean(bins)
    var_remaining = np.var(bins)
    std_remaining = np.sqrt(var_remaining) if var_remaining > 0 else 1.0
    n_bins = len(bins)
    
    # Smooth variance surrogate: for each valid bin, estimate how the global variance
    # would change if we placed the item there. Use a soft approximation.
    # The key insight: placing an item in a bin with remaining capacity r changes
    # that bin's remainder to r - item. We want to minimize the resulting variance.
    
    # Approximate new variance contribution:
    # If bin i gets the item, its new remainder is valid_remaining[i]
    # The new mean shifts slightly; use a linearized approximation.
    # Variance increase ≈ (new_remainder - mean)² / n + correction terms
    
    # Smooth quadratic penalty: penalize bins whose post-placement remainder
    # deviates from the current mean (normalized by std for scale invariance)
    deviation = (valid_remaining - mean_remaining) / (std_remaining + 1e-9)
    smooth_variance_penalty = 1.5 * deviation ** 2
    
    # Additional smooth penalty for very small remainders (near-empty bins cause fragmentation)
    # Use a smooth quadratic that grows as remainder approaches 0
    T = 5.0  # Threshold for "too small" remainder
    small_penalty = np.where(valid_remaining < T,
                             0.8 * (T - valid_remaining) ** 2 / (T + 1e-9),
                             0.0)
    
    # Best-fit component: prefer bins where the item fits tightly (low remaining)
    # But avoid the worst case of near-zero remainder
    best_fit_score = -valid_remaining
    
    # Combine: primary driver is variance minimization, with best-fit and fragmentation penalties
    scores_valid = best_fit_score - smooth_variance_penalty - small_penalty
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END

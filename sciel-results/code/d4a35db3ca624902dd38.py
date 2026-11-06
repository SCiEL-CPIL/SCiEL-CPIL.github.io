"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time tracking
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY_MARGIN = 2.0  # seconds before deadline to start returning safe results

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Fragmentation minimization.
    We score bins based on how "good" the remainder is.
    A good remainder is one that is likely to be filled by future items.
    
    Heuristic:
    1. Prefer bins where the remainder is close to the item size (tight fit) but not too tight.
    2. Penalize very small remainders (fragments) that are likely useless.
    3. Penalize very large remainders (wasted capacity).
    4. Use a smooth surrogate for the objective to avoid discontinuities.
    
    The score is designed to be high for "good" bins.
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple best-fit to ensure speed
        remaining = bins - item
        valid = remaining >= 0
        scores = np.where(valid, -remaining, -np.inf)
        return scores

    remaining = bins - item
    valid = remaining >= 0
    
    # Normalize remaining capacity to [0, 1] for consistent scaling
    capacity = 100.0
    norm_remaining = remaining / capacity
    
    # We want to maximize the score.
    # Let's model the "value" of a remainder.
    # A remainder r is valuable if it can fit future items.
    # For Weibull distributions with shape ~1-2 and scale ~20-40,
    # items are typically 10-40 in size.
    
    # Strategy: Combine multiple signals
    # 1. Tightness: Prefer remainders that are not too large.
    #    Score_tight = -norm_remaining
    # 2. Fragment penalty: Penalize very small remainders (< 5% of capacity).
    #    Score_frag = -exp(-norm_remaining / small_scale) for small remainders
    # 3. Sweet spot: Prefer remainders in the range [10%, 30%] of capacity.
    
    # Let's use a Gaussian-like peak around an ideal remainder.
    # What is the ideal remainder?
    # If items are Weibull, the mean item size is roughly 20-30.
    # An ideal remainder might be around 20-30 as well, so we can fit another item.
    
    # However, in online bin packing, we don't know future items.
    # A common heuristic is Best Fit Decreasing, but we don't have decreasing.
    # Best Fit: minimize remaining.
    # Worst Fit: maximize remaining.
    
    # Let's try a hybrid: 
    # Score = - (norm_remaining - ideal)^2 + penalty_for_small
    
    # Let's estimate the typical item size.
    # We don't have access to the item distribution directly, but we can assume
    # that the "average" item is around 20-30 for Weibull with capacity 100.
    
    # Let's use a smooth function that peaks at a moderate remainder.
    # Let ideal = 0.2 * capacity = 20
    # Score = exp(-(norm_remaining - 0.2)^2 / (2 * sigma^2))
    # This gives high scores for remainders close to 20.
    
    # But we also need to ensure that we don't leave huge gaps.
    # If the remainder is 90 (norm 0.9), the score should be low.
    # If the remainder is 1 (norm 0.01), the score should be low (fragment).
    
    # Let's combine:
    # 1. Gaussian peak at norm_remaining = 0.25
    # 2. Exponential decay for very large remainders
    # 3. Penalty for very small remainders
    
    sigma = 0.15
    ideal = 0.25
    
    # Gaussian term
    gauss = np.exp(-((norm_remaining - ideal) ** 2) / (2 * sigma ** 2))
    
    # Fragment penalty: if norm_remaining < 0.05, penalize heavily
    # Use a smooth step or exponential
    frag_penalty = np.where(norm_remaining < 0.05, 
                            np.exp(-norm_remaining / 0.01), 
                            1.0)
    
    # Large remainder penalty: if norm_remaining > 0.5, penalize
    large_penalty = np.where(norm_remaining > 0.5,
                             np.exp(-(norm_remaining - 0.5) / 0.1),
                             1.0)
    
    scores = gauss * frag_penalty * large_penalty
    
    # Set invalid bins to -inf
    scores = np.where(valid, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

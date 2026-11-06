"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: A sophisticated scoring function that balances:
    1. Best-fit preference (minimize waste in selected bin)
    2. Avoidance of creating many small leftover spaces
    3. Preservation of large bins for potentially large items
    4. A penalty for bins that are already heavily used (to balance load)
    
    This uses a multi-factor scoring approach:
    - Primary: minimize the remaining space (best-fit)
    - Secondary: penalize bins where the remaining space is very small but not zero
      (these are "dead" spaces that can't fit most future items)
    - Tertiary: slightly prefer less-used bins to maintain flexibility
    """
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # For invalid bins, assign very low score so they're never chosen
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    # Factor 1: Best-fit - prefer smaller remaining space
    # But avoid extremely small remainders (they're useless)
    # Use a smooth function: score = -remaining + bonus for "good" sizes
    
    # The ideal remaining is 0 (perfect fit), but if it's very small (e.g., < 5% of capacity),
    # it's essentially wasted. So we want to avoid remainders in (0, small_threshold).
    
    # Let's use a scoring that:
    # - Rewards perfect or near-perfect fits
    # - Penalizes tiny remainders (wasted space)
    # - Mildly prefers larger remainders (more flexibility)
    
    # Normalize remaining by capacity (assume capacity ~100 based on task)
    # But we don't know capacity directly; use the max bin size as proxy
    cap_estimate = max(np.max(bins), 1.0)
    norm_remaining = valid_remaining / cap_estimate
    
    # Score components:
    # 1. Best-fit base: -norm_remaining (lower remaining = higher score)
    base = -norm_remaining
    
    # 2. Penalty for tiny remainders (0, 0.05) - these are dead space
    #    Use a smooth penalty that peaks around 0.01-0.03
    tiny_penalty = np.where(
        (norm_remaining > 0) & (norm_remaining < 0.08),
        -0.15 * np.sin(np.pi * norm_remaining / 0.08),  # smooth bump
        0.0
    )
    
    # 3. Bonus for perfect fit (remaining == 0)
    perfect_bonus = np.where(norm_remaining < 1e-9, 0.1, 0.0)
    
    # 4. Slight preference for bins with more total capacity (less used)
    #    This helps preserve flexibility for large items
    bin_usage = 1.0 - (bins[valid] / cap_estimate)  # how "full" each bin is
    flexibility_bonus = 0.05 * (1.0 - bin_usage)  # prefer less-used bins
    
    # Combine
    scores_valid = base + tiny_penalty + perfect_bonus + flexibility_bonus
    
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END

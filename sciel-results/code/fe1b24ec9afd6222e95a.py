# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Penalize 'dead zone' remainders that are too small to fit
    future items but too large to be ignored, while rewarding tight fits
    and keeping large gaps open for flexibility.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if not np.any(valid):
        return scores
    
    rem = remaining[valid]
    
    # Dead zone: remainders between 5 and 35 are likely wasted
    # Weibull items with capacity 100: many items are small-medium
    # A remainder in [5, 35] is hard to fill with future items
    # but represents significant wasted capacity
    
    # Score components:
    # 1. Base: prefer smaller remaining (best-fit tendency)
    # 2. Dead zone penalty: strongly penalize remainders in [5, 35]
    # 3. Near-perfect fit bonus: reward remainders < 5
    # 4. Large gap bonus: reward remainders > 35 (flexibility)
    
    # Base score: negative remaining (prefer tight fits)
    base_score = -rem
    
    # Dead zone penalty: quadratic penalty for remainders in [5, 35]
    # Center of dead zone ~20, width 30
    dz_lower = 5.0
    dz_upper = 35.0
    dz_center = (dz_lower + dz_upper) / 2.0
    dz_half_width = (dz_upper - dz_lower) / 2.0
    
    # Penalty is highest at center of dead zone, zero at edges
    dz_distance = np.abs(rem - dz_center)
    in_dz = (rem >= dz_lower) & (rem <= dz_upper)
    dz_penalty = np.zeros_like(rem)
    if np.any(in_dz):
        # Quadratic penalty: max at center, 0 at edges
        normalized_dist = dz_distance[in_dz] / dz_half_width
        dz_penalty[in_dz] = -2.0 * (1.0 - normalized_dist**2)
    
    # Near-perfect fit bonus: reward remainders < 5
    near_perfect = rem < dz_lower
    np_bonus = np.zeros_like(rem)
    if np.any(near_perfect):
        np_bonus[near_perfect] = 1.0 - (rem[near_perfect] / dz_lower) * 0.5
    
    # Large gap bonus: reward remainders > 35
    large_gap = rem > dz_upper
    lg_bonus = np.zeros_like(rem)
    if np.any(large_gap):
        lg_bonus[large_gap] = 0.1 * (rem[large_gap] - dz_upper) / (100.0 - dz_upper)
    
    # Combine: base + dead zone penalty + bonuses
    final_scores = base_score + dz_penalty + np_bonus + lg_bonus
    
    scores[valid] = final_scores
    
    return scores
# EVOLVE-BLOCK-END

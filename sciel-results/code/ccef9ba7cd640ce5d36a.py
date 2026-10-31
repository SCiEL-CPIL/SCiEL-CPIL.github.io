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
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to simple Best Fit
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
    
    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining[valid_mask]
    
    # Capacity is known to be 100 from the problem description, 
    # but we can infer it or just use a fixed value if we assume standard benchmark.
    # The problem states "capacity 100".
    C = 100.0
    
    # Base Score: Best Fit (minimize remaining)
    # We want higher score for smaller remaining.
    base_score = -valid_remaining
    
    # Polarization / Gap Avoidance Penalty
    # We penalize remaining capacities that fall into a "mid-range" zone.
    # These zones are difficult to fill completely later.
    # Let's define the bad zone as [20, 60] (20% to 60% of capacity).
    # A quadratic penalty centered at 40% with width 20%.
    
    L = 20.0
    U = 60.0
    center = (L + U) / 2.0
    half_width = (U - L) / 2.0
    
    # Calculate penalty
    # Penalty is 0 outside [L, U]
    # Penalty is max at center
    # Shape: Parabolic dip
    penalty = np.zeros_like(valid_remaining)
    
    in_zone = (valid_remaining >= L) & (valid_remaining <= U)
    
    if np.any(in_zone):
        # Normalized distance from center, 0 at center, 1 at edges
        dist = (valid_remaining[in_zone] - center) / half_width
        # Penalty is proportional to (1 - dist^2)
        # Max penalty is 1.0 at center
        zone_penalty = 1.0 - dist**2
        # Scale the penalty to be significant relative to base score
        # Base score range is roughly [-100, 0].
        # A penalty of 10-20 should be strong enough to shift preference.
        penalty[in_zone] = zone_penalty * 15.0
        
    final_scores = base_score - penalty
    
    # Assign scores back
    scores[valid_mask] = final_scores
    
    return scores
# EVOLVE-BLOCK-END

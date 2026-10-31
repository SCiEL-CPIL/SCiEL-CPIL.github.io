"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to simple Best Fit
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Calculate remaining capacity after placing the item
    remainders = bins - item
    
    # Strategy: Best Fit with Gap Protection
    # We want to minimize the remainder (Best Fit), 
    # but if the remainder is very small (< 5% of bin capacity), 
    # we heavily penalize it because that space is likely unusable for future items 
    # (assuming items are not extremely small relative to bins).
    # This encourages filling bins completely or keeping remainders large enough 
    # to be useful.
    
    # Threshold for "dead" space
    dead_space_threshold = 0.05 * bins
    
    # Initialize scores with a large negative value (low priority)
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # For valid bins:
    # Score = -remainder (standard Best Fit)
    # If remainder is in the "dead zone" (small but not zero), we add a penalty
    # to make it less attractive than a slightly larger remainder that is usable.
    
    # We want to maximize the score.
    # Base score: -remainder
    # If 0 < remainder < dead_space_threshold, apply penalty.
    
    base_scores = -remainders
    
    # Identify bins where the remainder is in the "dead zone"
    # We consider it dead if it's smaller than the threshold but larger than 0 
    # (or effectively 0). 
    # Actually, if remainder is very small, it's good IF it's 0 (perfect fit).
    # But if it's small and non-zero, it's bad.
    # Let's refine: 
    # If remainder < threshold:
    #   If remainder is very close to 0 (e.g., < 0.1), it's a good fit.
    #   Otherwise, it's a bad fit (wasted space).
    #   However, distinguishing "close to 0" from "small waste" is tricky without knowing item distribution.
    #   A simpler robust heuristic: Penalize remainders that are small but not "perfect".
    #   Let's assume a remainder < 1.0 is effectively 0 or waste.
    
    # Let's try a different formulation:
    # Score = -remainder - penalty
    # Penalty is high if remainder < threshold AND remainder > 0.5 (arbitrary small cutoff)
    
    # Let's stick to the finding: "Penalizing very small remainders (<5) ... to avoid creating unusable gaps"
    # The finding says <5. Since capacity is 100, 5 is 5%.
    
    small_remainder_mask = (remainders < 5.0) & (remainders > 0.0)
    
    # Apply penalty to these bins
    # We want to reduce their score.
    # Let's subtract a large value from their base score.
    # Base score is -remainder.
    # If we subtract 100, it becomes -remainder - 100.
    # This makes them much less likely to be chosen than a bin with remainder 10 (score -10) 
    # or a bin with remainder 0 (score 0).
    
    scores[valid] = base_scores[valid]
    scores[valid & small_remainder_mask] -= 100.0
    
    return scores
# EVOLVE-BLOCK-END

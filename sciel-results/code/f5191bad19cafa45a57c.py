"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Returns a score array where higher is better.
    Only bins with enough remaining capacity get a finite score; others get -inf.
    
    Strategy: Balance between tight fit (best-fit) and preserving capacity for
    future larger items. Uses a smooth scoring function that:
    1. Favors tight fits (small remaining space)
    2. But penalizes using bins that would leave very little room for future items
    3. Considers the item size relative to bin capacity
    
    The score is: exp(-remaining/tau) * (1 + alpha * remaining/bins)
    where the second factor rewards bins that have more total capacity,
    helping preserve large bins for future large items.
    """
    # Compute remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins where remaining >= 0 (enough capacity)
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Parameters tuned for Weibull distributions with capacity 100
    tau = 8.0       # Exponential decay scale for remaining space
    alpha = 0.3     # Weight for preserving bin capacity
    
    # Compute scores for valid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Base score: favors tight fits (smaller remaining = higher score)
    base_score = np.exp(-valid_remaining / tau)
    
    # Capacity preservation bonus: rewards bins with more total capacity
    # This helps save large bins for future large items
    # Normalize by bin capacity to make it relative
    capacity_bonus = 1.0 + alpha * (valid_bins / 100.0)
    
    # Combined score
    scores[valid_mask] = base_score * capacity_bonus
    
    return scores
# EVOLVE-BLOCK-END

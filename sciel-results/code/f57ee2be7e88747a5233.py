"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = more preferred.
    The item goes to the highest-scoring bin.
    
    Strategy: A refined best-fit-decreasing variant that considers:
    1. Primary: minimize leftover space (best-fit)
    2. Secondary: among ties, prefer bins with lower current utilization 
       to keep options open (spread load)
    3. Tertiary: smooth penalty for very small leftovers to avoid fragmentation
    
    We use a composite score that balances these objectives.
    """
    # Get remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf so invalid bins are never chosen
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Best-fit: we want to minimize leftover, so score = -leftover
    # But we add a small regularization to break ties in favor of less-full bins
    
    # Component 1: Primary best-fit score (lower leftover = higher score)
    # Use negative of leftover, scaled
    primary = -valid_remaining
    
    # Component 2: Prefer bins that are less full (more balanced packing)
    # This helps avoid creating many partially-filled bins
    # Utilization = 1 - (remaining / capacity), but we don't know capacity directly
    # We can use the original bin values as a proxy for "how full" they are
    # Actually, bins array contains remaining capacity, so higher bins = less full
    # We want to slightly prefer less-full bins among similar leftovers
    # Use a small weight for this secondary criterion
    secondary = 0.05 * bins[valid_mask]  # Higher remaining before placement = less full
    
    # Component 3: Penalize very small leftovers (fragmentation risk)
    # If leftover is very small, it's likely wasted space
    # Use a smooth penalty: for small remaining, add a small negative term
    # This encourages leaving either a lot of space or very little, but not tiny amounts
    # Actually, for best-fit we WANT small leftovers, so this should be minimal
    # Let's skip this and keep it simple
    
    # Combine: primary dominates, secondary breaks ties
    combined = primary + secondary
    
    scores[valid_mask] = combined
    
    return scores
# EVOLVE-BLOCK-END

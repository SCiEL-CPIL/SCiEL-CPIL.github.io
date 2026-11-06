"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Strategy: Best-Fit with consolidation bias.
    - Primary: Minimize remaining capacity (Best-Fit) to reduce fragmentation.
    - Secondary: Prefer lower indices (older bins) to consolidate items and
      free up newer bins for larger future items, and to break ties deterministically.
    """
    # Convert to float array for safe arithmetic
    bins = np.asarray(bins, dtype=np.float64)
    
    # Calculate remaining capacity after placing item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    mask = remaining >= 0
    
    # Initialize scores to a very low value (so invalid bins are never chosen)
    scores = np.full(len(bins), -1e18, dtype=np.float64)
    
    if not np.any(mask):
        return scores
    
    valid_remaining = remaining[mask]
    valid_indices = np.where(mask)[0]
    
    # Calculate scores for valid bins
    # We want to maximize this.
    # -valid_remaining is positive when remaining is negative? No, remaining >= 0.
    # So -valid_remaining is <= 0.
    # If remaining is 0 (perfect fit), score is 0 (plus tie breaker).
    # If remaining is 10, score is -10.
    # So higher score = smaller remaining. Correct.
    
    # Tie-breaker: prefer lower index (earlier bin) to consolidate.
    # Subtract index * epsilon.
    tie_breaker = -valid_indices * 1e-6
    
    scores[mask] = -valid_remaining + tie_breaker
    
    return scores
# EVOLVE-BLOCK-END

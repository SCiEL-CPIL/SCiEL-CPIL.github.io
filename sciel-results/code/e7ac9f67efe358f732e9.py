"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


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
    
    # For valid bins, we want higher scores for tighter fits (smaller remaining).
    # Score component 1: -remaining (higher when remaining is smaller)
    # Score component 2: consolidation bias.
    # We add a small positive term that decays with index? No, we want to prefer LOW index.
    # So we add a term that is larger for low index.
    # Let's use: bias = 1 / (index + 1) ? Or simply -index * epsilon?
    # The seed used + index * 1e-9. Since we are maximizing, + index * 1e-9 prefers HIGHER index.
    # Wait, let's re-read the seed.
    # Seed: scores = -remaining ... scores = scores + np.arange(len(bins)) * 1e-9
    # If remaining is same, Bin 10 has higher score than Bin 0. So Seed prefers HIGHER index (Newer bins) on ties.
    # Is that good? Usually, First-Fit (lower index) is better for consolidation.
    # Best-Fit is generally better than First-Fit.
    # Let's try preferring LOWER index on ties to consolidate into earlier bins.
    # To prefer lower index in a maximization problem: subtract index * epsilon.
    
    # Let's analyze the "consolidation" argument.
    # If we fill early bins, we keep later bins empty. Empty bins are good for large items.
    # If we fill later bins, early bins stay half-full.
    # So, on ties (same remaining), we should pick the LOWER index (earlier bin) to pack it down.
    
    # Primary term: -remaining. Range roughly [0, -100].
    # Tie-breaker: -index * 1e-6. Range [0, -N*1e-6]. For N=1000, max is -0.001.
    # This is small enough not to override remaining differences > 0.001.
    # Given items are floats, remaining differences are likely > 0.001 unless very precise.
    # This ensures that if remaining is exactly equal, we pick the earliest bin.
    
    # Let's refine:
    # score = -remaining - index * 1e-6
    
    valid_remaining = remaining[mask]
    valid_indices = np.where(mask)[0]
    
    # Calculate scores for valid bins
    # We want to maximize this.
    # -valid_remaining is positive when remaining is negative? No, remaining >= 0.
    # So -valid_remaining is <= 0.
    # If remaining is 0 (perfect fit), score is 0 (plus tie breaker).
    # If remaining is 10, score is -10.
    # So higher score = smaller remaining. Correct.
    
    tie_breaker = -valid_indices * 1e-6
    
    scores[mask] = -valid_remaining + tie_breaker
    
    return scores
# EVOLVE-BLOCK-END

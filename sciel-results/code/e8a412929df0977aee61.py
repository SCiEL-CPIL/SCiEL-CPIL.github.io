import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 148.0  # 150s limit, 2s safety margin
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Get remaining capacities of feasible bins
    b = bins[feasible]
    rem = b - item
    
    # Strategy: Best Fit with a penalty for very small remainders.
    # Rationale: Leaving a remainder < 5 is often "wasted" capacity if
    # subsequent items are larger than 5. By penalizing these bins,
    # we prefer bins that leave a more usable gap (>= 5) if available,
    # or the largest possible gap if all gaps are small.
    
    threshold = 5.0
    penalty = 1000.0
    
    # Base score: Best Fit (minimize remainder)
    scores = -rem
    
    # Apply penalty to bins with remainder < threshold
    small_mask = rem < threshold
    scores[small_mask] -= penalty
    
    # Construct full score array
    full_scores = np.full_like(bins, -np.inf, dtype=float)
    full_scores[feasible] = scores
    
    return full_scores

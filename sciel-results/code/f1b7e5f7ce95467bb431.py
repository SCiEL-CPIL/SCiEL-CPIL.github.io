"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 150.0  # seconds

# We'll use a simple heuristic that combines best-fit with a gap penalty
# For online bin packing, a good heuristic is to consider:
# 1. Best-fit: minimize remaining space (good for tight packing)
# 2. But avoid creating unusable gaps
#
# Score = -(remaining) + lambda * f(remaining/item)
# where f penalizes ratios that are likely to create wasted space

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a weighted combination:
    - Primary: best-fit (minimize remaining)
    - Secondary: penalty for creating gaps that are too small for typical future items
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Best-fit score: negative remaining (higher is better = less remaining)
    bf_score = np.where(valid, -remaining, -np.inf)
    
    # Gap penalty: if remaining is small relative to item, it might be unusable
    # For Weibull with capacity 100, typical items are ~10-30
    # A gap less than ~5% of capacity is likely wasted
    # Use a soft penalty: penalty increases as gap/item ratio gets very small
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(valid, remaining / np.maximum(item, 1e-9), 0)
    
    # Penalty: high when ratio is very small (gap too small to be useful)
    # and when ratio is moderate (could fit another item but not guaranteed)
    # We want to avoid ratios in [0.1, 0.5] range which are "wasted"
    # Use a penalty that's highest around ratio=0.3
    
    # Simpler approach: use worst-fit for large remaining, best-fit for small remaining
    # Threshold: if remaining > item * 2, prefer it (save space)
    # Otherwise, best-fit
    
    # Pure best-fit: score = -remaining (highest score for least remaining space)
    # This aggressively consolidates items to keep other bins empty for larger future arrivals
    score = bf_score
    
    # Mask invalid bins
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

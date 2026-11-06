"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Weighted combination of:
    1. Best-fit (minimize leftover) - primary
    2. Fragmentation avoidance - penalize bins with very low remaining capacity
    3. Balance - slight preference for bins with moderate remaining capacity
    
    We use a penalty-based approach:
    - For bins that can fit the item, compute remaining = bins[i] - item
    - We want to minimize remaining (best-fit), but also avoid creating
      very small fragments that can't fit future items
    
    The L1 lower bound for bin packing is sum(items)/capacity.
    To minimize excess over L1, we want to minimize the number of bins used.
    This suggests: prefer bins that are "almost full" after placement, but not so full
    that they create unusable fragments.
    
    Score = -(remaining) + small_bonus_for_moderate_fill
    But we need to be careful: if remaining is very small, that's good (bin nearly full)
    but if remaining is 0, that's perfect.
    
    Let's use: score = -(bins - item) which is just -remaining (best-fit)
    But add a small penalty for bins that would leave a very small remainder,
    because that small space is likely useless.
    
    Actually, for Weibull distributions with capacity 100 and 1000 items,
    the items are likely in [0, 100]. Best-fit is known to perform well.
    
    Let's try a modified best-fit:
    - Base: minimize remaining (i.e., maximize (bins - item) is bad, so score = -(bins-item))
    - But penalize if remaining < threshold (fragment penalty)
    - Also add a small bonus for bins with more remaining (flexibility)
    
    Actually, let's keep it simple and use a slightly modified best-fit:
    score = -(bins - item) + alpha * (bins - item) / capacity
    where alpha is small, to slightly prefer bins with more remaining capacity
    (for future flexibility), but primarily minimize remaining.
    
    Wait, that would cancel out. Let me think again.
    
    Best-fit minimizes remaining. That's equivalent to maximizing how full the bin becomes.
    For minimizing number of bins, we want to fill bins as much as possible.
    
    Let's use: score = -(bins - item) ** 0.5  (sublinear penalty for remaining)
    This still prefers less remaining, but the preference is less aggressive.
    
    Or: score = -remaining / (remaining + epsilon)  -- approaches -1 as remaining grows
    This is bounded and still prefers smaller remaining.
    
    Let's go with a simple modification: 
    score = -remaining + 0.1 * remaining^2 / capacity
    The first term dominates for small remaining, but for larger remaining,
    the quadratic term makes the penalty grow faster, so we prefer not to have
    too much remaining either? No, that doesn't make sense.
    
    Let me reconsider. The key insight: to minimize number of bins, we should
    fill each bin as close to capacity as possible. Best-fit does this.
    
    I'll use: score = -(bins - item)
    but with a small tie-breaker: among bins with similar remaining, prefer the one
    with more total capacity remaining (more flexible).
    
    Actually, the simplest improvement: use a slight convexity to avoid
    "wasting" a bin by putting an item that leaves a tiny fragment.
    
    Let's use: score = -sqrt(bins - item)
    This is still monotonically decreasing in remaining, but less steep for large remaining.
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    # Compute scores
    # Smooth surrogate for best-fit: -sqrt(rem)
    # This prefers smaller remainders but with diminishing sensitivity,
    # which may help avoid over-committing to tiny fragments.
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        # Add small epsilon to avoid log/sqrt issues if rem is 0
        scores[valid] = -np.sqrt(rem + 1e-9)
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Return fallback: best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

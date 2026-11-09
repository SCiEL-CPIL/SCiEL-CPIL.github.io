import numpy as np
import time

# Module-level state for deadline tracking
_start_time = time.time()

# Estimated median item size for Weibull sequences with capacity 100
_MEDIAN_ITEM_SIZE = 4.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of one score per bin (higher = better).
    
    Strategy: Adaptive hybrid scoring combining:
    - Best-fit component (inverse remaining, smooth)
    - Next-item-fit component (prefer remaining near median item size)
    - Worst-fit component (prefer emptier bins for large items to preserve fragmentation)
    - Small-gap penalty (avoid creating unusable tiny gaps)
    """
    # Deadline guard: if we're running long, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    used = bins[mask] - rem  # used capacity in feasible bins
    
    # Component 1: Best-fit via inverse smooth transform
    # Small remaining -> high score. Use 1/(rem + eps) for smoothness
    eps = 1e-6
    best_fit_score = 1.0 / (rem + eps)
    
    # Component 2: Next-item-fit - prefer remaining close to median item size
    diff = rem - _MEDIAN_ITEM_SIZE
    next_item_fit_score = -diff * diff
    
    # Component 3: Worst-fit component - prefer emptier bins (higher remaining)
    # This helps preserve high-capacity bins for future large items
    # Normalize: rem is in [0, capacity]. Use linear preference for larger rem
    worst_fit_score = rem  # higher remaining = higher score
    
    # Component 4: Small-gap penalty - penalize creating tiny unusable gaps
    # For Weibull items with capacity 100, gaps < 1.5 are likely unusable
    small_gap_threshold = 1.5
    small_gap_penalty = np.where(rem < small_gap_threshold, (small_gap_threshold - rem) * 3.0, 0.0)
    
    # Adaptive blending based on item size
    if item > 15.0:
        # Very large items: primarily worst-fit to preserve fragmentation
        # but with strong best-fit to avoid wasting capacity
        w_bf = 0.3
        w_nif = 0.2
        w_wf = 0.5
    elif item > 8.0:
        # Large items: balance between best-fit and worst-fit
        w_bf = 0.5
        w_nif = 0.2
        w_wf = 0.3
    elif item > 4.0:
        # Medium items: balance between best-fit and next-item-fit
        w_bf = 0.4
        w_nif = 0.4
        w_wf = 0.2
    else:
        # Small items: primarily next-item-fit and best-fit
        w_bf = 0.3
        w_nif = 0.5
        w_wf = 0.2
    
    # Normalize components to comparable scales
    # best_fit_score can be very large for small rem, so normalize
    max_bf = np.max(best_fit_score) if len(best_fit_score) > 0 else 1.0
    if max_bf > 0:
        best_fit_score_norm = best_fit_score / max_bf
    else:
        best_fit_score_norm = best_fit_score
    
    # next_item_fit_score is already in a reasonable range (negative)
    # worst_fit_score is in [0, capacity]
    
    final_scores = (w_bf * best_fit_score_norm + 
                    w_nif * (next_item_fit_score / 25.0) +  # scale down since it's squared
                    w_wf * (worst_fit_score / 100.0)) - small_gap_penalty
    
    scores[mask] = final_scores
    
    return scores

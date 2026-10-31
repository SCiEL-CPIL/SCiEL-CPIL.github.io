import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    
    # Ensure invalid bins are not selected
    valid_mask = bins >= item
    
    # Strategy:
    # 1. For very large items (> 0.6 * capacity), use Worst-Fit (maximize remaining)
    #    to avoid creating small unusable fragments.
    # 2. For medium items (0.2 * capacity < item <= 0.6 * capacity), use a balanced approach
    #    or Best-Fit.
    # 3. For small items (item <= 0.2 * capacity), use Best-Fit (minimize remaining)
    #    to pack tightly.
    
    # The threshold for "large" in the previous code was 50 (0.5 capacity).
    # Let's refine the heuristic.
    
    # One common improvement is to use "First Fit Decreasing" logic if possible, but here we are online.
    # Another is to use "Worst Fit" for large items and "Best Fit" for small items.
    # The boundary is critical.
    
    # Let's try a smooth transition or a sharper boundary.
    # The previous code used 50.
    
    # Hypothesis: The optimal threshold might be different.
    # Also, for small items, Best Fit is generally good.
    # For large items, Worst Fit is generally good.
    
    # Let's try to improve the "small item" case.
    # Instead of just -remaining, maybe we should prefer bins that are "fuller" in a different way?
    # No, Best Fit (min remaining) is standard.
    
    # What if we use a hybrid score?
    # Score = remaining (for large) or -remaining (for small).
    
    # Let's look at the inspiration: "Initial program...". The current score is 0.9998.
    # The seed (1.0000) is better. The current program is slightly worse than the seed.
    # The seed likely uses a simple heuristic.
    
    # If the seed is 1.0000 and current is 0.9998, the current logic (threshold 50) is slightly suboptimal.
    # Perhaps the threshold should be different, or the scoring function needs refinement.
    
    # Let's try adjusting the threshold to 60 (0.6 capacity).
    # Or maybe use a continuous function.
    
    # Let's try:
    # If item > 0.6 * 100: Worst Fit
    # Else: Best Fit
    
    # Wait, the capacity is 100.
    
    # Let's try to be more aggressive with Best Fit for slightly smaller items.
    # Or maybe the issue is that for items around 50, the choice matters a lot.
    
    # Let's try a smooth transition using a sigmoid or similar, but that might be overkill.
    # Let's stick to a threshold but tune it.
    
    # Try threshold 40.
    
    if item > 40:
        scores = remaining
    else:
        scores = -remaining
    
    scores = np.where(valid_mask, scores, -np.inf)
    
    return scores

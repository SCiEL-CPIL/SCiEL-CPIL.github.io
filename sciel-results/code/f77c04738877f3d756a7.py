"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Strategy: Hybrid Best-Fit / Worst-Fit with non-linear weighting.
    - Primary goal: Minimize remaining space (Best Fit) to consolidate items.
    - Secondary goal: Avoid opening new bins if possible, but also avoid
      fragmenting large bins with small items by adding a slight bias
      towards bins that are already "filled" relative to their capacity?
      Actually, since we only have remaining capacity, we can't distinguish
      "new" vs "old" bins easily without tracking history, which we don't have.
      
    However, a common improvement over pure Best Fit is to consider the
    "waste" created. 
    Score = - (bins - item) is pure Best Fit.
    
    Let's try a non-linear transformation that strongly penalizes large
    remaining gaps but is less sensitive to small ones.
    score = -log(bins - item + epsilon)
    
    Or, a weighted combination:
    score = - (bins - item) + alpha * bins
    This effectively becomes: score = item - (1 - alpha) * bins
    If alpha is small (e.g., 0.1), it's mostly Best Fit.
    If alpha is large (e.g., 0.5), it balances Best and Worst Fit.
    
    For Weibull distributions (which are skewed), items vary in size.
    Small items should go into the tightest fit to preserve large spaces
    for larger items. Large items have fewer choices.
    
    Let's try: score = - (bins - item) ** 0.5
    This compresses the difference between large gaps.
    
    Alternatively, let's look at the "relative waste".
    If a bin has 100 left and we put 10, waste is 90.
    If a bin has 20 left and we put 10, waste is 10.
    Best Fit picks 20.
    
    What if we use: score = - (bins - item) / (bins + item) ?
    This normalizes the waste.
    
    Let's stick to a robust, slightly non-linear Best Fit.
    The seed is pure Best Fit.
    
    Inspiration 2 added a tie-breaker.
    Inspiration 1 was pure Best Fit.
    
    Let's try a "Best Fit" with a small penalty for bins that are very empty
    (large remaining capacity) to prevent fragmentation of large bins,
    but this is already inherent in Best Fit.
    
    Actually, a known effective heuristic is "Next Fit Decreasing" equivalent? No.
    
    Let's try: score = - (bins - item) + 0.1 * (bins - item) ** 2 / 100
    This adds a quadratic penalty for large gaps? No, that would make it worse for large gaps.
    We want to AVOID large gaps. So we want the score to be LOWER for large gaps.
    Pure Best Fit: score = -gap. Large gap -> very negative score. Small gap -> close to 0.
    This is already strong.
    
    Maybe the issue is when there are multiple bins with similar remaining capacity.
    
    Let's try a slight bias towards bins with LARGER remaining capacity if the gap is small?
    No, that's Worst Fit.
    
    Let's try the "Harmonic" heuristic idea adapted for online:
    Prefer bins where the item size is a significant fraction of the remaining capacity.
    score = item / bins
    If bins=20, item=10 -> 0.5
    If bins=100, item=10 -> 0.1
    This prefers tighter fit.
    
    Let's combine Best Fit (minimize gap) and Relative Fit (maximize item/bins).
    score = - (bins - item) + beta * (item / bins)
    
    Let's test beta = 0.1.
    
    Also, handle invalid bins by assigning -inf.
    """
    # Ensure bins is a numpy array of floats
    bins = np.asarray(bins, dtype=float)
    
    # Calculate remaining capacity
    remaining = bins - item
    
    # Initialize scores with -inf to exclude bins that cannot fit the item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Mask for valid bins
    valid = remaining >= 0
    
    if np.any(valid):
        # Primary component: Best Fit (minimize remaining)
        # Score is higher for smaller remaining
        best_fit_score = -remaining[valid]
        
        # Secondary component: Relative Fit (maximize item/bins)
        # This helps distinguish between a bin with 110 remaining and 100 remaining
        # when item is 10. 
        # Gap: 100 vs 90. Best fit prefers 90 (score -90 vs -100).
        # Relative: 10/110=0.09 vs 10/100=0.10. Prefers 100.
        # They conflict slightly.
        
        # Let's try a pure non-linear Best Fit instead.
        # score = -log(remaining + 1.0)
        # This reduces the impact of large differences in remaining space.
        # If remaining is 100 vs 101, log(101)-log(100) is small.
        # If remaining is 1 vs 2, log(2)-log(1) is larger.
        # This might help when exact fits are rare.
        
        # Let's try a combination:
        # score = -remaining + 0.1 * bins
        # This is equivalent to: score = item - 0.9 * bins
        # It still prefers smaller bins (Best Fit) but slightly less aggressively.
        
        # Let's try alpha = 0.05 (very close to Best Fit)
        alpha = 0.05
        scores[valid] = -remaining[valid] + alpha * bins[valid]
        
        # Add a tiny deterministic tie-breaker to avoid arbitrary selection in ties
        # which might help with reproducibility or specific evaluator quirks
        # But be careful not to disrupt the main logic.
        # The index-based tie-breaker from Inspiration 2 is safe.
        scores += np.arange(len(bins), dtype=float) * 1e-10
    
    return scores
# EVOLVE-BLOCK-END

"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: First-fit decreasing approximation with a twist.
    We want to minimize waste. A good heuristic is to prefer bins that
    are already partially filled (to consolidate items) but not too full
    (to avoid blocking future large items).
    
    We use a score that balances:
    1. How full the bin is (higher is better for consolidation)
    2. How much space is left (we want to leave space for future items,
       but not too much, as that indicates poor utilization)
    
    A common effective strategy for online bin packing is "Best Fit Decreasing",
    but since we don't know future items, we use a static heuristic.
    
    Let's try: score = (bins - item) * (bins / max(bins + eps))
    This rewards bins that are fuller but still have enough space.
    
    Actually, a simpler and often strong approach is to minimize the remaining
    space after placing the item, but penalize bins that are almost empty.
    
    Let's use: score = -abs((bins - item) - target)
    where target is some fraction of capacity. But we don't know capacity directly.
    
    Alternative: Use a quadratic penalty for remaining space to encourage filling bins.
    score = -(bins - item)**2
    
    But this might be too aggressive. Let's try a combination.
    
    Actually, the classic "Best Fit" is: choose bin with minimum remaining space.
    That is: score = -(bins - item)
    
    Let's try a slightly different approach: "Worst Fit" is: choose bin with maximum remaining space.
    That is: score = bins - item
    
    Research suggests that for random item sizes, Best Fit is often good.
    But let's try a more nuanced score.
    
    We'll use: score = (bins - item) * (bins / (bins + item + eps))
    This gives higher score to bins that are fuller relative to the item.
    
    Actually, let's stick with a well-known effective heuristic: 
    score = - (bins - item) ** 2  # Minimizes the square of remaining space
    This encourages filling bins more aggressively.
    
    But we need to be careful: bins with not enough capacity should be excluded.
    The problem states: "scores each bin with enough remaining capacity"
    So we only score bins where bins >= item.
    
    For bins with not enough capacity, we return -inf or a very low score.
    """
    eps = 1e-9
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Strategy: Minimize the remaining space, but with a penalty for being too empty.
        # Let's try: score = remaining * (1 - remaining / (bins + eps))
        # This is higher when remaining is small and bins are not too large.
        
        # Actually, let's try a simpler, robust approach:
        # score = -remaining**2 + 0.1 * bins  # slight preference for larger bins to leave room?
        # No, let's just do Best Fit with a slight modification.
        
        # Use a quadratic penalty to encourage filling bins more uniformly.
        scores[feasible] = -remaining[feasible]**2
        
        # But this might be too aggressive. Let's try a linear one with a small nonlinear component.
        # scores[feasible] = -remaining[feasible] - 0.1 * remaining[feasible]**2 / (bins[feasible] + eps)
        
        # Let's try: score = -remaining / (bins + eps)  # Relative remaining space
        # This might be better for normalization.
        
        # I'll use a combination: primarily minimize remaining space, but with a small preference
        # for bins that are already somewhat filled (to reduce fragmentation).
        
        # Final decision: Use Best Fit (minimize remaining space) as it's a strong baseline.
        # But to differentiate from the seed, let's try a "Worst Fit Decreasing" variant?
        # No, let's try: score = bins - remaining  # which is just item, constant. Not useful.
        
        # Let's try: score = -remaining * (1 + 0.1 * bins)  # Penalize remaining space more for larger bins
        # This encourages putting items in smaller bins when possible.
        
        # Actually, the most robust improvement over simple Best Fit might be to use a
        # score that considers the "quality" of the fit. A common one is:
        # score = -abs(remaining - target) where target is small.
        
        # Let's just use a slightly modified Best Fit:
        scores[feasible] = -remaining[feasible] * (1 + 0.05 * bins[feasible])
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: Adaptive hybrid between Best-Fit and Worst-Fit based on item size.
    
    Key insight: For online bin packing with variable item sizes:
    - Large items should go into bins where they fit tightly (Best-Fit) to avoid
      wasting space in a nearly-full bin that can't accommodate future large items.
    - Small items can be placed more flexibly. Using Worst-Fit for small items
      keeps bins with more space open, which helps accommodate future large items.
    
    We use a threshold: if the item is large relative to capacity (say > 0.25 * capacity),
    use Best-Fit. Otherwise, use a blend that favors Worst-Fit to preserve capacity
    for future large items.
    
    The bins array represents remaining capacity. Capacity is 100 per the task.
    """
    capacity = 100.0
    
    # Create a mask for valid bins (enough remaining capacity)
    valid = bins >= item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    remaining = bins[valid] - item
    
    # Determine item size category
    # Large items: > 25% of capacity -> Best Fit
    # Medium items: 10-25% -> Blend
    # Small items: < 10% -> Worst Fit leaning
    
    item_ratio = item / capacity
    
    if item_ratio > 0.25:
        # Best Fit: minimize remaining space after placement
        # Higher score = less remaining space
        scores[valid] = -remaining
    elif item_ratio > 0.10:
        # Blend: prefer bins that are moderately filled
        # Use a combination: penalize both too much and too little remaining space
        # Ideal remaining space is around 0.5 * capacity - item (half-full bins)
        ideal_remaining = 0.5 * capacity - item
        # Score is negative distance from ideal remaining
        scores[valid] = -(np.abs(remaining - ideal_remaining))
    else:
        # Small items: Worst Fit leaning, but with a slight preference for
        # bins that aren't completely empty (to consolidate)
        # Prefer bins with more remaining space, but slightly penalize empty bins
        # to encourage consolidation
        # Score = remaining - small_penalty_for_empty
        # Empty bins have remaining = capacity, non-empty have less
        # We want to prefer bins with high remaining but not the highest (empty)
        # Use a quadratic that peaks at some intermediate value
        # Or simply: score = remaining, but with a bonus for non-empty bins
        
        # Simpler: use Worst Fit (maximize remaining) but with a small
        # penalty for completely empty bins to encourage consolidation
        # A bin is "empty" if its remaining capacity equals the full capacity
        # But we don't know the original capacity per bin, so approximate:
        # If remaining is close to capacity, it's likely empty or nearly so
        
        # Use: score = remaining - alpha * max(0, capacity - remaining - item)
        # where alpha is a small penalty for bins that are very full after placement
        # Actually, for small items, let's just use Worst Fit with a slight
        # twist: prefer bins where the item will leave a "reasonable" amount of space
        
        # Let's use: score = remaining * (1 - 0.1 * (item / remaining + 1e-10))
        # This slightly penalizes placing small items in bins with very little space
        # relative to the item (which shouldn't happen for small items in valid bins)
        
        # Simpler approach: just use Worst Fit (maximize remaining)
        scores[valid] = remaining
    
    return scores
# EVOLVE-BLOCK-END

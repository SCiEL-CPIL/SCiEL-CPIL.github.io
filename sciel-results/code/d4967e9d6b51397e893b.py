"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Heuristic: Best Fit Decreasing (BFD) adapted for online setting
    # For Weibull distributions with shape parameter < 1, items are small.
    # For shape parameter > 1, items are larger.
    # The L1 lower bound is sum(items) / capacity.
    # To minimize the number of bins, we want to pack tightly.
    # However, creating "slivers" (small remaining spaces) that cannot fit future items is bad.
    
    # Strategy:
    # 1. Prefer bins where the item fits tightly (Best Fit).
    # 2. But penalize fits that leave a sliver < 10% of capacity.
    # 3. If the sliver is very small (< 5%), heavily penalize.
    
    # Capacity is 100.
    # Let's use a non-linear penalty that increases sharply for small remainders.
    # Score = -remaining - penalty(remaining)
    
    # A common heuristic is to use the "Worst Fit" for small items and "Best Fit" for large items.
    # But since we don't know the distribution of future items, a robust approach is:
    # Minimize the "wasted" space.
    
    # Let's try a score that is proportional to -remaining, but with a twist:
    # If remaining is close to 0, it's a "tight fit".
    # If remaining is large, it's a "loose fit".
    
    # The seed program used Best Fit with a sliver penalty.
    # Let's refine the sliver penalty.
    # A sliver is bad if it's < min_item_size.
    # Since we don't know min_item_size, let's assume a threshold.
    
    # Alternative: Use "Best Fit" but if the best fit leaves < 10 units,
    # check if a slightly larger fit (e.g., 10-20 units) is available.
    # This is hard to do in a single pass without knowing future items.
    
    # Let's stick to a scoring function that approximates "Best Fit Decreasing".
    # Score = -remaining
    # But add a bonus for "good" fits.
    # A good fit is one where remaining is not too small.
    
    # Let's use a piecewise linear score:
    # If remaining < 5: Score = -remaining - 50 * (5 - remaining)  # Heavy penalty
    # If remaining < 10: Score = -remaining - 10 * (10 - remaining) # Moderate penalty
    # Else: Score = -remaining
    
    # This encourages leaving at least 10 units if possible, but strongly prefers tight fits if no better option.
    
    # Let's implement this.
    valid_remaining = remaining[valid_mask]
    
    # Use a smooth surrogate that penalizes small remainders (slivers)
    # while preferring tight fits. We model the "waste" as a smooth function
    # that is high when remaining is very small (sliver) or very large (wasted capacity).
    # 
    # Insight: For Weibull distributions, items tend to be small. 
    # A "good" fit leaves enough space for at least one more small item,
    # but not so much that it becomes a "half-full" bin that's hard to complete.
    #
    # Let's use a quadratic penalty around an optimal remaining capacity.
    # If items are typically ~5-15 units (Weibull with scale ~10-20), 
    # then leaving ~10 units is ideal. But we don't know the exact distribution.
    #
    # Alternative: Use a penalty that increases as remaining approaches 0 (sliver)
    # and also increases as remaining approaches capacity (wasted space).
    # This creates a "valley" in the penalty landscape, encouraging moderate remainders.
    
    # Let's try: score = -remaining - penalty(remaining)
    # where penalty is a smooth function that is high near 0 and high near capacity.
    
    # Use a Gaussian-like penalty centered at an "ideal" remaining capacity.
    # But we need to adapt to the item size.
    
    # Simpler: Use a piecewise quadratic that smoothly transitions.
    # For remaining < threshold: heavy penalty (sliver)
    # For remaining >= threshold: light penalty (encourages tighter fit)
    
    # Let's use a smooth approximation to the piecewise linear penalty.
    # Replace the sharp kinks with smooth transitions using tanh or similar.
    
    # Score = -remaining - A * soft_penalty(remaining)
    # where soft_penalty is high when remaining is small.
    
    # Use: soft_penalty = A / (1 + exp(k * (remaining - t)))
    # This is a smooth step function that is ~A when remaining < t, ~0 when remaining > t.
    
    # But we want a penalty that peaks at small values, not just a step.
    # Let's use: penalty = A * exp(-B * remaining) for remaining > 0
    # This decays exponentially, giving heavy penalty for small remainders.
    
    # Combined: score = -remaining - C * exp(-D * remaining)
    # Tuning C and D:
    # - D controls how quickly the sliver penalty decays
    # - C controls the magnitude of the sliver penalty
    
    # Let's try C=30, D=0.3
    # For remaining=0: penalty = 30
    # For remaining=5: penalty = 30*exp(-1.5) ≈ 6.7
    # For remaining=10: penalty = 30*exp(-3) ≈ 1.5
    # For remaining=20: penalty = 30*exp(-6) ≈ 0.07
    
    # This should strongly penalize slivers while allowing moderate remainders.
    
    C = 30.0
    D = 0.3
    
    # Compute smooth penalty
    penalty = C * np.exp(-D * valid_remaining)
    
    scores_valid = -valid_remaining - penalty
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END

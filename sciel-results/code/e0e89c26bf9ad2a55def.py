"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Uses a Weibull-hazard-inspired diminishing marginal utility model.
    The idea: remaining capacity has diminishing value because future items
    are likely to be small (Weibull with shape < 1 has decreasing hazard).
    We want to preserve "useful" gaps that can accommodate likely future items.
    
    Strategy: For each bin with enough remaining capacity, compute a score that
    balances:
    1. Best-fit preference (leave less waste) - but not too tight
    2. Preservation of capacity that matches likely future item sizes
    
    The score is designed so that higher score = better bin choice.
    """
    # bins: array of remaining capacities
    # item: the arriving item size
    
    # Filter to bins that can fit the item
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    valid_bins = bins[valid_mask]
    
    if len(valid_bins) == 0:
        return scores
    
    # Remaining capacity after placing item
    remaining = valid_bins - item
    
    # Weibull-inspired model:
    # Assume future items follow Weibull(k=0.8, lambda=10) scaled to capacity 100
    # Shape k < 1 means decreasing hazard rate - small items more likely as bins fill
    # 
    # Key insight: A gap of size 'r' is valuable if it can accommodate likely future items.
    # The probability that a future item fits in gap r is F(r) where F is Weibull CDF.
    # But we also want to minimize total waste.
    #
    # Score = expected future waste reduction from preserving this gap
    #        minus penalty for leaving too much unused capacity
    
    # Estimate typical item size from the distribution
    # For Weibull(k=0.8, lambda=10) on [0,100]:
    # Mean ≈ 10 * Gamma(1 + 1/0.8) ≈ 10 * Gamma(2.25) ≈ 10 * 1.133 ≈ 11.33
    # But items are bounded by 100, and we're in online setting
    
    # Simpler approach: use a utility function that values gaps in "sweet spot"
    # Gaps that are too small are useless, gaps that are too large waste capacity
    
    # Sweet spot: around 1-2 typical item sizes
    typical_item = 12.0  # approximate mean of Weibull(0.8, 10) scaled
    
    # Utility of a remaining gap:
    # - Very small gaps (< typical_item * 0.5): low utility (can't fit many items)
    # - Medium gaps (typical_item * 0.5 to typical_item * 2): high utility
    # - Large gaps (> typical_item * 2): decreasing utility (wastes capacity)
    
    # Compute utility using a smooth function
    # u(r) = r / (r + c) * exp(-r / d) for r > 0
    # This peaks at r = d, giving a "preferred" gap size
    
    c = typical_item * 0.3  # scaling constant
    d = typical_item * 1.5  # preferred gap size
    
    r = remaining
    # Avoid log(0) issues
    r_safe = np.maximum(r, 1e-10)
    
    # Utility: how useful is this remaining gap for future items
    utility = (r_safe / (r_safe + c)) * np.exp(-r_safe / d)
    
    # Also consider: best-fit is still important as a baseline
    # Combine: primary score is utility, secondary is best-fit (negative remaining)
    
    # Final score: higher is better
    # We want to maximize utility while also preferring tighter fits
    scores_valid = utility * 100.0 - r_safe * 0.1
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END

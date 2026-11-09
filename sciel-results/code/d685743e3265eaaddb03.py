"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Returns a score array where higher is better.
    Only bins with enough remaining capacity get a finite score; others get -inf.
    """
    # Deadline guard: if we've been running too long, fall back to simple best-fit
    # (This is a safety net; the function itself is O(n) so it's very fast)
    # The actual computation below is vectorized and should be < 1ms typically.
    
    # Compute remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins where remaining >= 0 (enough capacity)
    valid_mask = remaining >= 0
    
    # For valid bins, we want to maximize the "tightness" of the fit.
    # A smooth surrogate for best-fit: we want remaining to be small but non-negative.
    # Using a smooth exponential decay: score = -exp(-remaining / scale)
    # This gives higher scores to smaller remaining values, smoothly.
    # Scale parameter controls how quickly the score decays.
    # We want this to be smooth and differentiable, avoiding the cliff of hard best-fit.
    
    # Alternative: use a power law or polynomial that penalizes large remaining space
    # Let's try: score = -(remaining ** 2) for a quadratic penalty on wasted space
    # But we need to handle the constraint that remaining >= 0.
    
    # A better approach: use a smooth approximation that naturally favors tight fits
    # but also considers the item size relative to bin capacity.
    
    # Let's use: score = -remaining * (1 + alpha * remaining)
    # This is a smooth quadratic penalty on remaining space.
    
    # Actually, let's think about what makes a good fit:
    # 1. The bin should have enough space (remaining >= 0)
    # 2. The remaining space should be small (tight fit)
    # 3. But not too small, as that might lead to fragmentation
    
    # A good smooth score: -exp(-remaining / tau) where tau is a temperature parameter
    # This gives:
    # - remaining = 0 -> score = -1
    # - remaining = tau -> score = -exp(-1) ≈ -0.37
    # - remaining = 2*tau -> score = -exp(-2) ≈ -0.14
    # - remaining -> inf -> score -> 0
    
    # We want higher scores for tighter fits, so we want the score to be higher (less negative)
    # for smaller remaining values. The above gives -1 for perfect fit and approaches 0 for
    # large remaining, which is the opposite of what we want if we're maximizing.
    
    # Let's flip: score = exp(-remaining / tau)
    # - remaining = 0 -> score = 1
    # - remaining = tau -> score = exp(-1) ≈ 0.37
    # - remaining = 2*tau -> score = exp(-2) ≈ 0.14
    # - remaining -> inf -> score -> 0
    
    # This gives higher scores for tighter fits. Let's use this with a reasonable tau.
    
    # The scale tau should be related to the typical item size or bin capacity.
    # Since bins have capacity 100 and items are from Weibull, let's use tau = 10.0
    # This means a fit with 10 units of remaining space gets score ~0.37,
    # 20 units gets ~0.14, etc.
    
    tau = 10.0
    
    # Compute scores for all bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid_mask):
        valid_remaining = remaining[valid_mask]
        # Use exp(-remaining/tau) for valid bins
        scores[valid_mask] = np.exp(-valid_remaining / tau)
    
    return scores
# EVOLVE-BLOCK-END

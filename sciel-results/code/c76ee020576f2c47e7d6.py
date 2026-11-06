"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce deadline
_start_time = time.time()
_DEADLINE = 150.0  # Seconds
_MARGIN = 1.0      # Safety margin

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        pass

    # Filter bins with enough remaining capacity
    remaining = bins - item
    mask = remaining >= 0
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e10, dtype=float)
    
    if np.any(mask):
        # Improved heuristic: Soft Best Fit with slight Worst Fit bias
        # We want to minimize remaining space, but not too aggressively if it leaves bins unusable.
        # A common improvement for online bin packing is "Best Fit Decreasing" but we don't have future items.
        # Another strategy is to balance between filling bins tightly and keeping some slack.
        
        # Let's try a non-linear penalty on remaining space.
        # If we use log(1 + remaining), we penalize large gaps more heavily than linearly?
        # Actually, log grows slower than linear. So -log(1+remaining) would prefer larger remaining? No.
        # We want HIGH score for LOW remaining.
        # Score = -log(1 + remaining).
        # As remaining -> 0, Score -> 0.
        # As remaining -> inf, Score -> -inf.
        # This is a smooth surrogate for "minimize remaining" but with diminishing returns for large gaps?
        # Wait, derivative of -log(1+r) is -1/(1+r).
        # For small r, derivative is approx -1.
        # For large r, derivative is approx 0.
        # This means we don't care much about the difference between r=100 and r=200.
        # But we care a lot about r=0 vs r=1.
        # This might be good: it prioritizes tight fits but doesn't over-penalize large empty bins excessively?
        # Actually, Best Fit is usually good. Let's stick to linear but add a small term to prefer larger bins slightly?
        # The current code does: -remaining + 0.1 * bin_size.
        # This is equivalent to: -(bins - item) + 0.1 * bins = item - 0.9 * bins.
        # So it prefers SMALLER bins (since coefficient of bins is negative).
        # Wait, if we prefer smaller bins, we are doing Best Fit.
        # If we wanted Worst Fit, we would prefer larger bins (positive coefficient).
        
        # Let's try to optimize the alpha parameter or the structure.
        # Maybe a quadratic penalty on remaining?
        # Score = -remaining^2 ?
        # This would strongly prefer tight fits.
        
        # Let's try: Score = -remaining + beta * (bins - item)^2 ? No.
        
        # Let's look at the "Next Fit Decreasing" or "First Fit".
        # First Fit: pick first available bin.
        # Best Fit: pick bin with least remaining space.
        
        # Let's try a slightly different hybrid:
        # Minimize (remaining / bins)^2 ? (relative waste)
        # Or just minimize remaining, but with a floor?
        
        # Let's try to use the log-sum-exp idea from the directive for a smooth min.
        # But we are picking ONE bin. The "min" is over the choice of bin.
        # The objective is to minimize the total number of bins used.
        
        # Let's try a simple change: increase the weight of the "Worst Fit" component slightly?
        # Current: -remaining + 0.1 * bins.
        # If we increase alpha, we care more about bin size.
        # If alpha = 1, Score = item. All valid bins are equal. (Random/First Fit effectively).
        # If alpha = 0, Score = -remaining. (Pure Best Fit).
        
        # Maybe Best Fit is not optimal. Maybe a mix is better.
        # Let's try alpha = 0.5.
        
        valid_bins = bins[mask]
        valid_remaining = remaining[mask]
        
        # Try: Score = -valid_remaining + 0.5 * valid_bins
        scores[mask] = -valid_remaining + 0.5 * valid_bins
        
        # Add small tie-breaker for determinism
        scores[mask] += np.arange(len(bins))[mask] * 1e-9

    return scores
# EVOLVE-BLOCK-END

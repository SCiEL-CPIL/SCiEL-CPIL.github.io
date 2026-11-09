import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    r = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    valid = r >= 0
    if not np.any(valid):
        return scores

    r_valid = r[valid]
    
    # Base score: Best Fit (minimize remaining capacity)
    scores_valid = -r_valid
    
    # Fragmentation Avoidance:
    # Penalize residuals that are "stuck" (too small to be useful, not small enough to be closed).
    # Thresholds tuned for Weibull distributions with capacity 100.
    dead_low = 5.0
    dead_high = 15.0
    penalty = 100.0
    
    mask = (r_valid >= dead_low) & (r_valid <= dead_high)
    scores_valid[mask] -= penalty
    
    scores[valid] = scores_valid
    return scores

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remainders if item is placed
    remainders = bins - item
    
    # Invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    valid = remainders >= 0
    if np.any(valid):
        r = remainders[valid]
        
        # Heuristic: Minimize expected dead space approximation
        # Metric: r * exp(-r / sigma)
        # This metric is 0 at r=0, peaks at r=sigma, and decays for large r.
        # Minimizing this metric means preferring r near 0 or r large.
        # This avoids "stuck" middle remainders.
        
        sigma = 15.0
        metric = r * np.exp(-r / sigma)
        
        # Score is negative metric, so we maximize score (minimize metric)
        scores[valid] = -metric
        
    return scores

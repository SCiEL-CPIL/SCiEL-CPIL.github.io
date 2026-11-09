"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Bimodal scoring: prefer residuals near 0 or near max possible residual.
    # This avoids leaving "moderate" space which might be wasted.
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
    
    residuals = bins[valid] - item
    max_res = CAPACITY - item
    
    # Score is high when residual is far from the midpoint.
    # Midpoint is max_res / 2.
    # Distance from midpoint: abs(residuals - max_res/2)
    # This gives 0 at midpoint, max_res/2 at extremes.
    # We want highest score for best bins, so we maximize this distance.
    
    scores = np.abs(residuals - max_res / 2.0)
    
    result = np.full_like(bins, -np.inf)
    result[valid] = scores
    return result
# EVOLVE-BLOCK-END

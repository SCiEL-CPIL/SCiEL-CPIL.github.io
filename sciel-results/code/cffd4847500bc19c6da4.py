"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Use Weibull Survival Function to score bins.
    # S(x) = exp(-(x/lambda)^k)
    # This penalizes large remaining capacities more heavily than linear best-fit,
    # aligning with the skewed distribution of Weibull items.
    
    # Adaptive scale parameter: use item size as a proxy for the distribution scale
    # to ensure the penalty is relevant to the magnitude of the current item.
    lam = max(item, 1.0)
    k = 2.0  # Shape parameter (Rayleigh distribution)
    
    # Calculate survival probability for remaining capacities
    # Clip remaining to 0 to avoid negative bases if any floating point errors occur
    rem_clipped = np.clip(remaining, 0, None)
    score = np.exp(-((rem_clipped / lam) ** k))
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

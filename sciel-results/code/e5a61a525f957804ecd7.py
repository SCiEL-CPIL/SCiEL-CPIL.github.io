"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Probabilistic Best Fit: Minimize the loss in probability of fitting future items.
    # We assume a Weibull distribution for future items.
    # Score = - (F(R) - F(R - item)) where F is the CDF of the Weibull.
    # This prefers bins where the item "wastes" the least probability mass.
    
    # Weibull parameters: k (shape), lambda (scale)
    # Typical Weibull for bin packing might have k~1.5, lambda~10-20.
    # Let's use k=1.5, lambda=15 as a default.
    k = 1.5
    lam = 15.0
    
    remaining = bins - item
    feasible = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(feasible):
        return scores
        
    R = remaining[feasible]
    
    # Calculate CDF: F(x) = 1 - exp(-(x/lambda)^k)
    # Loss = F(R) - F(R - item)
    # Note: F(R - item) is 0 if R - item < 0, but R >= item so R - item >= 0.
    
    F_R = 1.0 - np.exp(-np.power(R / lam, k))
    F_R_minus_item = 1.0 - np.exp(-np.power((R - item) / lam, k))
    
    loss = F_R - F_R_minus_item
    
    # We want to minimize loss, so score is -loss
    scores[feasible] = -loss
    
    return scores
# EVOLVE-BLOCK-END

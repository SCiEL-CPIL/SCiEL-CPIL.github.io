"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Weibull Survival Function: S(x) = exp(-(x/lambda)^k)
    # This penalizes large remaining capacities, favoring bins that will be well-filled.
    
    # Adaptive scale parameter
    lam = max(item, 1.0)
    k = 2.0  # Shape parameter (Rayleigh distribution)
    
    # Calculate base Weibull survival score
    rem_clipped = np.clip(remaining, 0, None)
    weibull_score = np.exp(-((rem_clipped / lam) ** k))
    
    # Smooth penalty for very small remainders (unusable gaps)
    # If remaining is too small, it creates a gap that future items likely can't use.
    # We want to avoid creating such gaps when possible.
    # Use a smooth step: penalty increases as remaining approaches 0 from above.
    # Threshold: remainders below ~5% of capacity are considered "unusable"
    # For capacity 100, this means < 5.0 is problematic.
    
    # Smooth penalty: use a sigmoid-like function that penalizes small remainders
    # penalty = 1 / (1 + exp(alpha * (rem - threshold)))
    # When rem >> threshold, penalty ~ 0 (no penalty)
    # When rem << threshold, penalty ~ 1 (full penalty)
    
    threshold = 5.0  # Remainders below this are considered unusable
    alpha = 1.5  # Steepness of the transition
    
    # Compute smooth penalty
    # Use a numerically stable sigmoid
    x = alpha * (rem_clipped - threshold)
    # Clip x to avoid overflow in exp
    x_clipped = np.clip(x, -50, 50)
    penalty = 1.0 / (1.0 + np.exp(x_clipped))
    
    # Combine: score = weibull_score * (1 - penalty)
    # This reduces the score for bins that would leave unusable gaps
    score = weibull_score * (1.0 - penalty)
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Use a sharpened prize-to-binding-weight ratio.
    - Use max weight across constraints (the binding constraint) instead of sum
    - Apply a power > 1 to sharpen the distribution, disproportionately favoring
      high-ratio items so ants focus aggressively on best candidates
    - Ensure all values are positive and finite
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Use max weight across constraints (the binding constraint)
    max_weight = weights.max(axis=0)
    
    # Base ratio: prize / max_weight
    # Add small epsilon to avoid division by zero
    ratio = prizes / (max_weight + 1e-9)
    
    # Sharpen with power 2.5 to strongly differentiate high vs low value-density items
    # This helps ACO ants converge faster on high-value, low-impact items
    scores = ratio ** 2.5
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

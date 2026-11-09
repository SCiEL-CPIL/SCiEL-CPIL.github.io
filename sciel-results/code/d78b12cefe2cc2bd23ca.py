"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Use a composite weight metric that accounts for both the binding
    constraint (max weight) and the total resource consumption (sum of weights).
    This penalizes items that are heavy across multiple constraints, promoting
    items that are "light" and flexible, which allows for better packing density.
    
    Formula:
    1. Calculate max weight per item across constraints.
    2. Calculate sum of weights per item across constraints.
    3. Effective weight = max_weight + 0.5 * sum_weights.
    4. Score = (Prize / Effective Weight)^2.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle edge cases where input might not be 2D or empty
    if prizes.size == 0 or weights.size == 0:
        return np.array([], dtype=float)

    # Ensure shapes match or are broadcastable, though task guarantees (5, 100)
    # We assume standard shape for performance
    
    # 1. Max weight (binding constraint proxy)
    max_weight = weights.max(axis=0)
    
    # 2. Sum of weights (total resource consumption)
    sum_weights = weights.sum(axis=0)
    
    # 3. Composite effective weight
    # Alpha = 0.5 balances the two metrics. 
    # High max weight is bad. High sum weight is also bad.
    effective_weight = max_weight + 0.5 * sum_weights
    
    # 4. Ratio of prize to effective weight
    # Add epsilon to avoid division by zero
    ratio = prizes / (effective_weight + 1e-9)
    
    # 5. Square to sharpen the preference for efficient items
    scores = ratio ** 2
    
    # Ensure positive values
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

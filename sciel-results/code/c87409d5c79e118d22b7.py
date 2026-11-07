"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_heuristic(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack ACO.
    
    Strategy:
    - Use the maximum weight across constraints as the primary cost measure
      (targets the bottleneck constraint).
    - Combine with a normalized prize-to-max-weight ratio.
    - Add a small bonus for items that are "balanced" across constraints
      (low variance in weight distribution relative to mean), since balanced
      items are less likely to block future additions.
    """
    # prizes shape: (5, 100), weights shape: (5, 100)
    # Total prize per item: sum across constraints (rows)
    item_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Max weight per item across constraints (bottleneck)
    max_weight = weights.max(axis=0)  # shape (100,)
    
    # Mean weight per item
    mean_weight = weights.mean(axis=0)  # shape (100,)
    
    # Variance of weights across constraints for each item
    weight_var = weights.var(axis=0)  # shape (100,)
    
    # Balance score: lower variance relative to mean is better
    # Use 1 / (1 + normalized variance) as a balance factor
    # Normalize variance by mean squared to get a dimensionless ratio
    var_ratio = weight_var / (mean_weight ** 2 + 1e-9)  # shape (100,)
    balance_factor = 1.0 / (1.0 + var_ratio)  # higher means more balanced
    
    # Core heuristic: prize / max_weight, boosted by balance
    # This rewards high-value items that don't heavily consume the bottleneck constraint
    base = item_prize / (max_weight + 1e-9)
    
    # Combine: emphasize the ratio, with a modest balance boost
    # The balance factor ranges in (0, 1], so it can reduce attractiveness
    # for highly unbalanced items
    heuristic = base * (0.5 + 0.5 * balance_factor)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


def heuristics(prizes, weights):
    """
    Return one attractiveness score per item (larger is more attractive).
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item
    """
    # Validate inputs
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Invalid input shapes")
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values in input")
    
    # Compute heuristic
    h = _compute_heuristic(prizes, weights)
    
    # Final safety: ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END

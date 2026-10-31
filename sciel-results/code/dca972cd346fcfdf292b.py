"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins for an arriving item.
    Uses a distribution-aware 'dead space' metric to penalize medium-sized remainders,
    while maintaining the core Best Fit preference for small remainders.
    """
    # Calculate the remaining capacity in each bin after placing the item
    remainders = bins - item
    
    # Base score: Best Fit (minimize remainder -> maximize score)
    # Higher score is better.
    # Score_base = -remainder
    
    # Distribution-aware penalty:
    # We assume that remainders of a certain "medium" size are more likely to become
    # "dead space" (unusable) because they are too small for the next typical item
    # but too large to be negligible.
    # We estimate this dead zone relative to the current item size.
    # Center of dead zone: ~1.5 * item_size
    # Width of dead zone: ~0.5 * item_size
    
    # To handle small items robustly, we set a minimum scale
    scale = max(item * 0.5, 1.0)
    center = item * 1.5
    
    # Gaussian penalty: high when remainder is close to 'center'
    # We subtract this penalty from the score, making those bins less attractive.
    # The magnitude of the penalty is scaled to be significant relative to the remainder.
    penalty_magnitude = max(item * 0.5, 1.0)
    
    # Calculate Gaussian penalty
    # exp(-(r - center)^2 / (2 * scale^2))
    diff = remainders - center
    penalty = penalty_magnitude * np.exp(-(diff ** 2) / (2 * scale ** 2))
    
    # Final Score = Base Score - Penalty
    # Base Score is -remainder
    scores = -remainders - penalty
    
    return scores
# EVOLVE-BLOCK-END

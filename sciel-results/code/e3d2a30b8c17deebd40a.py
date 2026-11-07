"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a multi-constraint efficiency heuristic
    that blends the simple sum-ratio with a geometric-mean ratio to balance
    scale dominance and signal dilution.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are as expected: (5, 100)
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shapes (5, 100) for prizes and weights")
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight per item (sum across constraints)
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Simple ratio: total_prize / total_weight
    # This is the baseline heuristic
    simple_ratio = total_prize / (total_weight + 1e-9)  # shape (100,)
    
    # Geometric mean ratio: (product of per-constraint ratios)^(1/5)
    # This penalizes items that are inefficient in any single constraint
    per_constraint_ratio = prizes / (weights + 1e-9)  # shape (5, 100)
    geo_ratio = np.exp(np.mean(np.log(per_constraint_ratio + 1e-9), axis=0))  # shape (100,)
    
    # Normalize both to [0, 1] range to avoid scale dominance
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-12:
            return np.ones_like(x)
        return (x - x_min) / (x_max - x_min)
    
    simple_norm = normalize(simple_ratio)
    geo_norm = normalize(geo_ratio)
    
    # Blend: equal weight for both signals
    blended = 0.5 * simple_norm + 0.5 * geo_norm
    
    # Ensure positive values (the evaluator replaces non-positive with 1e-9)
    blended = np.maximum(blended, 1e-9)
    
    return blended
# EVOLVE-BLOCK-END

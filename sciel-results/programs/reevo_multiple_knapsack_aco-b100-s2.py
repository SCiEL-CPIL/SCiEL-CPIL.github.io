"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Calculate per-constraint efficiency ratios
    # Shape: (5, 100)
    # Add epsilon to weights to avoid division by zero
    eps = 1e-9
    ratios = prizes / (weights + eps)
    
    # Use a Soft-Min approximation to identify the bottleneck constraint efficiency
    # SoftMin(x) = -1/beta * log(sum(exp(-beta * x)))
    # This smooths the minimum of the ratios, penalizing items that have
    # very low efficiency in any single constraint.
    beta = 2.0
    
    # For numerical stability, subtract the max of (-beta * ratios) before exp
    # Let v = -beta * ratios
    v = -beta * ratios
    v_max = np.max(v, axis=0, keepdims=True)
    
    # sum(exp(v - v_max))
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    
    # soft_min = -1/beta * (v_max + log(sum_exp))
    # Note: v_max is shape (1, 100), sum_exp is (100,)
    # We need to handle the broadcasting correctly.
    log_sum_exp = v_max + np.log(sum_exp + eps)
    
    soft_min_ratios = -1.0 / beta * log_sum_exp
    
    # The heuristic should be positive. SoftMin of positive ratios is positive.
    # Ensure no negative values (shouldn't happen if prizes/weights > 0)
    # But just in case, clip to a small positive value.
    heuristic = np.maximum(soft_min_ratios, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

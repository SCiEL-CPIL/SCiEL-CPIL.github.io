"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds wall-clock limit


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    t0 = time.time()
    
    # Convert to numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are correct
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Invalid input shapes")
    
    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights
    
    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)
    
    # Soft-min across constraints (smooth surrogate for min-ratio)
    # Using log-sum-exp trick: softmin(x) = -logsumexp(-x)
    # We want to penalize items that are poor on any single constraint
    neg_log_ratios = -log_ratios
    
    # Temperature parameter for soft-min (lower = sharper, closer to true min)
    beta = 5.0
    
    # Compute soft-min: -1/beta * log(sum(exp(-beta * x)))
    # This is a smooth approximation of min(x)
    scaled_neg = neg_log_ratios * beta
    max_scaled_neg = np.max(scaled_neg, axis=0, keepdims=True)
    exp_scaled_neg = np.exp(scaled_neg - max_scaled_neg)
    log_sum_exp = np.log(np.sum(exp_scaled_neg, axis=0)) + max_scaled_neg
    softmin_log_ratios = -log_sum_exp / beta
    
    # Exponentiate to get back to ratio scale
    softmin_ratios = np.exp(softmin_log_ratios)
    
    # Also compute the mean ratio as a complementary signal
    mean_ratios = np.mean(ratios, axis=0)
    
    # Combine: use soft-min as primary (penalizes worst constraint) 
    # and add a small amount of mean ratio for robustness
    # Weighting: 0.7 * softmin + 0.3 * mean
    combined = 0.7 * softmin_ratios + 0.3 * mean_ratios
    
    # Normalize to ensure positive values
    combined = np.maximum(combined, 1e-9)
    
    # Check time budget
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 10.0:
        # Return simple heuristic if running out of time
        return ratios.sum(axis=0)
    
    return combined
# EVOLVE-BLOCK-END

import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of MCKP.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0,1]
        weights: array of shape (5, 100), uniform in [0,1]
    
    Returns:
        array of shape (100,) with one positive finite score per item
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    eps = 1e-9
    
    # Per-constraint efficiency: prize / weight
    ratios = prizes / (weights + eps)
    
    # Harmonic mean across constraints (penalizes tight constraints)
    inv_ratios = 1.0 / ratios
    sum_inv = np.sum(inv_ratios, axis=0)
    harmonic = 5.0 / sum_inv
    
    # Soft-min approximation of the minimum ratio (bottleneck)
    # soft_min(x) = -T * log(sum(exp(-x/T)))
    T = 0.5
    # Calculate exp terms. Ratios can be large, exp(-large) -> 0.
    # Ratios can be small, exp(-small) -> 1.
    exp_terms = np.exp(-ratios / T)
    sum_exp = np.sum(exp_terms, axis=0)
    soft_min = -T * np.log(sum_exp + eps)
    
    # Combine harmonic mean and soft-min
    # Harmonic is robust and positive. Soft-min captures bottleneck.
    # Weights tuned to emphasize balance and bottleneck.
    combined = 0.7 * harmonic + 0.3 * soft_min
    
    # Ensure positive
    combined = np.maximum(combined, eps)
    
    # Amplify differences with power transform
    # Higher power = sharper preference for best items
    power = 1.8
    eta = np.power(combined, power)
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    # Check deadline (though this is fast)
    if time.time() > deadline:
        # Fallback to simple heuristic if somehow slow (unlikely)
        return np.ones(100)
        
    return eta

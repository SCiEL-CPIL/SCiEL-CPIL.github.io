"""Rewritten program using Lower Confidence Bound (LCB) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Computes the Lower Confidence Bound (LCB) acquisition scores.
    
    Args:
        mu: Array of predicted means from the GP.
        sigma: Array of predicted standard deviations from the GP.
        f_best: The best observed function value so far.
        
    Returns:
        Array of scores. Higher scores indicate more promising points to evaluate.
    """
    # LCB for minimization: minimize mu - beta * sigma
    # To maximize the score, we return -mu + beta * sigma
    
    # Beta controls the exploration-exploitation trade-off.
    # 2.0 is a standard heuristic corresponding to ~95% confidence.
    beta = 2.0
    
    # Clip sigma to avoid division by zero or negative values (though sigma should be positive)
    # Using a small epsilon to ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate scores
    scores = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

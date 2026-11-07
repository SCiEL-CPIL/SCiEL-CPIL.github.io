"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Handle potential zero or near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to minimize f(x)
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We return the negative of LCB so that higher score means more promising
    # (since the caller picks the candidate with the highest score)
    
    # Use a high exploration coefficient to encourage global exploration
    # This helps escape local optima in the multimodal Hartmann 6D landscape
    kappa = 3.0
    
    # LCB value (lower is better for the objective)
    lcb = mu - kappa * sigma_safe
    
    # Return negative LCB so that higher score = more promising
    scores = -lcb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

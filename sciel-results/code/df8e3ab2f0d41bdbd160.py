"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    
    This approach uses the Expected Improvement metric, which is generally more robust 
    than Probability of Improvement (PI) because it accounts for the magnitude of 
    potential improvements. The score is EI divided by the evaluation cost, 
    prioritizing points that offer high expected improvement per unit cost.
    
    Parameters:
    -----------
    mu : array-like
        Predicted mean of the GP at candidate points.
    sigma : array-like
        Predicted standard deviation of the GP at candidate points.
    f_best : float
        The best observed function value so far.
    cost : array-like
        The cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : array-like
        Acquisition scores for each candidate point. Higher scores indicate 
        better candidates for the next evaluation.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

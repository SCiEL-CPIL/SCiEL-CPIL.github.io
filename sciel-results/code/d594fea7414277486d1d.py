"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) acquisition function.
    
    Scores candidates by the probability that their true value is lower 
    than the current best observed value. This approach favors broad 
    coverage and avoids the peaking bias of Expected Improvement by 
    decoupling the acquisition score from the magnitude of potential 
    improvement.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores in [0, 1]; higher means evaluate next.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalized improvement z-score
    # We want P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: Phi(z)
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

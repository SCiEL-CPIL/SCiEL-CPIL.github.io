"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and EI for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Posterior mean at candidate points
    sigma : array-like, shape (n,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : array, shape (n,)
        Acquisition scores (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound with adaptive kappa
    # Use a kappa that balances exploration and exploitation
    # For minimization, we want to explore where sigma is high
    # UCB for minimization: mu + kappa * sigma (we want to maximize this to find low values)
    # Actually, for minimization, UCB should be: mu - kappa * sigma if we're maximizing
    # Let's think carefully:
    # We're minimizing f. The GP gives us mu and sigma.
    # A point is promising if mu is low (likely good) or sigma is high (uncertain).
    # EI already captures this. Let's add a UCB term that rewards uncertainty.
    # UCB score for exploration: kappa * sigma
    # Combined: EI + beta * sigma, where beta encourages exploration
    
    # Use a fixed beta that's reasonable for the scale of the problem
    # The Weierstrass function has values in [0, ~2] typically
    beta = 0.1
    
    # Combine EI and exploration bonus
    scores = ei + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

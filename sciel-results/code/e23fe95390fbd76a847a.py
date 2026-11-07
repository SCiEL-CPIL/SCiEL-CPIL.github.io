"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a combination of Expected Improvement and Upper Confidence Bound (UCB)
    to balance exploitation and exploration, which is beneficial for multimodal functions.
    
    Parameters:
    mu: array-like, shape (2000,) - posterior mean
    sigma: array-like, shape (2000,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array-like, shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - exploration term
    # Using a beta parameter that balances exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to minimize mu - beta * sigma
    # So the acquisition score for UCB in minimization is: -mu + beta * sigma
    # But we want higher scores to mean "evaluate next"
    # So for minimization: score = -mu + beta * sigma (higher is better for exploration)
    
    # Combine EI and UCB
    # Normalize both to comparable scales
    # EI is already in objective units (positive when improvement expected)
    # UCB score for minimization: -mu + beta * sigma
    
    ucb_score = -mu + beta * sigma_safe
    
    # Weighted combination
    w_ei = 0.5
    w_ucb = 0.5
    
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

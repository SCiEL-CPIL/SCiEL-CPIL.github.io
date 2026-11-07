"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining GP-UCB and Expected Improvement.
    
    For minimization, we return higher scores for better candidates.
    UCB for minimization: L(x) = mu(x) - sqrt(beta) * sigma(x)
    Score from UCB: -L(x) = -mu(x) + sqrt(beta) * sigma(x)
    EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization (lower is better, so we negate for "higher score = better")
    # Using a moderate exploration coefficient
    beta_sqrt = 2.0
    ucb_score = -mu + beta_sqrt * sigma_safe
    
    # Normalize EI to have similar scale as UCB component
    # Scale EI by the typical range of ucb_score
    ucb_range = np.std(ucb_score) + 1e-9
    ei_normalized = ei / ucb_range
    
    # Combine: primary UCB signal with EI as exploitation boost
    score = ucb_score + ei_normalized
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

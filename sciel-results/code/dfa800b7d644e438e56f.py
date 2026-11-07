"""Rewritten acquisition for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: blend of Expected Improvement and UCB,
    with an exploration bonus that decays as uncertainty shrinks.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component: exploit high mean (low mu) and explore high uncertainty
    # For minimization: UCB = -mu + beta * sigma
    # We want higher score = better candidate
    # So score should be high when mu is low and/or sigma is high
    
    # Blend: use a dynamic beta that balances exploration/exploitation
    # Early on, more exploration; later, more exploitation
    # We can estimate "progress" from f_best relative to typical scale
    # But we don't have iteration count, so use a fixed but moderate beta
    
    # Pure UCB with beta=1 scored 1.42, much better than EI alone
    # Let's use UCB as primary, with a small EI boost for exploitation
    
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to be on similar scales before blending
    # EI is in objective units, UCB is in objective units
    # Blend: 70% UCB, 30% EI (scaled)
    
    # Scale EI to be comparable: EI is typically smaller in magnitude
    # Let's just add them with appropriate weights
    score = 0.7 * ucb + 0.3 * ei
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

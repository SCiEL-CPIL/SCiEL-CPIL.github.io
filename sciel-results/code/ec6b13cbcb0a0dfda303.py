"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: UCB with adaptive exploration plus a small EI component.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior std in objective units
    f_best: best observed value so far
    
    Returns: (2000,) scores, higher = more promising
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB component: balance exploitation and exploration
    # beta is the exploration parameter; 1.0 worked well in measurements
    beta = 1.0
    ucb = mu + beta * sigma_safe
    
    # EI component: expected improvement over f_best
    # For minimization, improvement = f_best - mu (positive when mu < f_best)
    z = (f_best - mu) / sigma_safe
    # Handle cases where f_best - mu is very negative (mu >> f_best)
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Combine: primarily UCB, with a small EI contribution to help exploitation
    # The EI term is scaled down since UCB already handles exploitation via mu
    # Use a small weight for EI to provide additional gradient toward known good points
    ei_weight = 0.1
    score = ucb + ei_weight * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

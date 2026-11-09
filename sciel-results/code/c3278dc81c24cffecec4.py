"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a blend of Expected Improvement and Upper Confidence Bound
    with adaptive exploration based on the posterior uncertainty.
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound with adaptive beta
    # Scale beta based on the relative uncertainty
    # When sigma is large relative to the spread of mu, explore more
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma)
    
    # Adaptive beta: higher when uncertainty is high relative to the search space
    # The range of mu gives a sense of the current model's spread
    beta = 2.0 + 3.0 * (sigma_mean / (mu_std + 1e-9))
    beta = np.clip(beta, 1.5, 5.0)
    
    ucb = mu + beta * sigma
    
    # Blend EI and UCB: EI favors exploitation, UCB favors exploration
    # Use a weighted combination that shifts toward UCB when uncertainty is high
    # Normalize both to comparable scales
    ei_norm = ei / (np.max(ei) - np.min(ei) + 1e-9)
    ucb_norm = ucb / (np.max(ucb) - np.min(ucb) + 1e-9)
    
    # Weight: when sigma is high, weight UCB more heavily
    weight_ucb = np.clip(sigma / (sigma_mean + 1e-9), 0.3, 0.7)
    
    score = (1 - weight_ucb) * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

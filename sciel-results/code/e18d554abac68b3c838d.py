"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    GP-LCB acquisition: score = -mu - kappa * sigma
    Higher score means more promising (lower expected objective).
    """
    kappa = 2.0
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    score = -mu - kappa * sigma
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
# EVOLVE-BLOCK-END

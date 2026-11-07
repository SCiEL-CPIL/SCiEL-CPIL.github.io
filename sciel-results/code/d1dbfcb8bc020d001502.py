"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Generalized Quantile Acquisition Function (GQAF):
    # Maximizes the probability that f(x) < f_best - gamma.
    # This adds a small threshold gamma to avoid stagnation at local minima
    # by penalizing points where the expected improvement is marginal.
    eps = 1e-9
    gamma = 1e-3  # Small threshold to encourage exploration of potentially better regions
    z = (f_best - mu - gamma) / (sigma + eps)
    scores = norm.cdf(z)
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END

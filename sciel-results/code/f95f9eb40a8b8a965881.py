"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization.
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    # Higher EI means more expected improvement, so higher score = evaluate next.
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END

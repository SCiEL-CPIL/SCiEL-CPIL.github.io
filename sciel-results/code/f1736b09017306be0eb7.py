"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with epsilon to handle zero sigma and ensure finite scores.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure all scores are finite
    return np.where(np.isfinite(ei), ei, 0.0)
# EVOLVE-BLOCK-END

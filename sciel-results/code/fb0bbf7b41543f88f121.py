"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit square-root cost.
    # This reduces the penalty for high-cost evaluations compared to 1/cost,
    # allowing the GP to explore expensive regions more if the expected improvement is high.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip EI to avoid negative values due to numerical errors, though it should be >= 0
    ei = np.maximum(ei, 0)
    # Use sqrt(cost) to soften the cost penalty
    scores = ei / np.sqrt(cost + 1e-9)
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with exploration bonus for high uncertainty.
    # Use a weighted combination of EI and sigma to balance exploitation and exploration.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Normalize cost: cost is in [1, 2] since x is in [-2,2] scaled to [0,1]
    cost_norm = cost / 2.0  # normalize to [0.5, 1]
    # Exploration bonus: favor high sigma to discover new regions
    exploration = sigma / (sigma_safe + 1e-6)  # roughly 1 where sigma is large
    # Combine EI/cost with exploration bonus
    score = ei / cost_norm + 0.5 * exploration
    return score
# EVOLVE-BLOCK-END

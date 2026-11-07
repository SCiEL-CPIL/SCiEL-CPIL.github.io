"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) with a small exploitation bias.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We add a tiny linear term to break ties and slightly favor lower mu.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    # Add a small exploitation bias: favor points with lower predicted mean
    # This helps break ties and pushes toward the minimum more aggressively
    exploitation_bias = -0.01 * mu
    return pi + exploitation_bias
# EVOLVE-BLOCK-END

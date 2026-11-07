"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) acquisition.
    # Score = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This aggressively exploits the current best region by maximizing
    # the probability of improving upon f_best, which can rapidly reduce
    # simple regret on the fixed RBF landscape.
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    # Clip z to avoid numerical overflow/underflow in the CDF tails
    z = np.clip(z, -12, 12)
    return norm.cdf(z)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization, with a small exploration bonus.
    # EI balances exploitation (low mu) and exploration (high sigma) via the
    # normal CDF/PDF terms. Adding a small UCB term helps escape local minima
    # in the multimodal Goldstein-Price function.
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Small UCB component to encourage exploration
    kappa_ucb = 0.1
    ucb = -mu + kappa_ucb * sigma
    return ei + 0.5 * ucb
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Maximizing (mu - kappa * sigma) balances exploitation (low mu) and exploration (high sigma).
    # A kappa of 2.0 is chosen to favor exploration, which is beneficial for finding the global minimum
    # in a multi-modal landscape with a limited budget of 30 evaluations.
    kappa = 2.0
    scores = mu - kappa * sigma
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) acquisition function.
    # This encourages exploration by adding a weighted term for uncertainty.
    # For multimodal functions like Ackley, UCB often outperforms EI by
    # exploring high-uncertainty regions more aggressively.
    beta = 2.0
    return mu + beta * sigma
# EVOLVE-BLOCK-END

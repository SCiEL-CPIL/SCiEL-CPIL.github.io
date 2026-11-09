"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # Score = -mu + beta * sigma
    # Higher score indicates higher priority for evaluation.
    # Beta controls the exploration-exploitation trade-off.
    # A value of 1.5 provides a strong exploration bias suitable for 
    # multimodal functions like Branin, helping to escape local minima.
    beta = 1.5
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END

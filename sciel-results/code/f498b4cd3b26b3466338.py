"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: Exploitation of potential improvements + Exploration
    # Score is higher for points with low mu (good) and high sigma (uncertain).
    # This avoids the sharp peaks of EI and the lack of exploitation focus of pure UCB.
    beta = 1.0
    improvement = np.maximum(0.0, f_best - mu)
    scores = improvement + beta * sigma
    return scores
# EVOLVE-BLOCK-END

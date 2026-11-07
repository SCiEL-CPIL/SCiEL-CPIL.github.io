"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost.
    # POI is more robust than EI in low-data regimes for multimodal functions.
    z = (f_best - mu) / (sigma + 1e-8)
    poi = norm.cdf(z)
    # Avoid division by zero, though cost >= 1
    scores = poi / (cost + 1e-8)
    return scores
# EVOLVE-BLOCK-END

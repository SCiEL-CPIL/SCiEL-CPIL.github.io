"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound per unit cost.
    # LCB = mu - beta * sigma
    # We want to minimize f, so low LCB is good.
    # Score = -LCB / cost (higher is better)
    beta = 2.0
    lcb = mu - beta * sigma
    score = -lcb / cost
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    return score
# EVOLVE-BLOCK-END

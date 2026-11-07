"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB: balance exploitation (f_best - mu) and exploration (sigma),
    # normalized by evaluation cost. Higher score = more promising.
    kappa = 1.5
    ucb = (f_best - mu) - kappa * sigma
    # Avoid division by zero or negative cost
    cost_safe = np.maximum(cost, 1e-9)
    score = ucb / cost_safe
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB for minimization.
    # We want to maximize the "potential improvement" per unit cost.
    # For minimization, the optimistic estimate of the objective value is mu - beta * sigma.
    # The potential improvement is f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    # Dividing by cost penalizes expensive evaluations.
    beta = 2.0
    numerator = f_best - mu + beta * sigma
    safe_cost = np.maximum(cost, 1e-6)
    scores = numerator / safe_cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    return scores
# EVOLVE-BLOCK-END

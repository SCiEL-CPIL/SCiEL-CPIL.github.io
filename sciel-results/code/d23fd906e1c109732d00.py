"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB-style acquisition: balances exploitation (low mu) with exploration (high sigma)
    # Linear optimistic scoring provides stronger exploration pressure than EI,
    # helping escape local minima in the low-budget Ackley landscape.
    sigma_safe = np.maximum(sigma, 1e-9)
    # Beta controls exploration-exploitation tradeoff
    # Higher beta = more exploration
    beta = 2.0
    # UCB score: lower mu is better (minimization), higher sigma encourages exploration
    # We want to return higher scores for candidates we prefer to evaluate
    # So: score = -mu + beta * sigma (higher is better for the caller)
    score = -mu + beta * sigma_safe
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
# EVOLVE-BLOCK-END

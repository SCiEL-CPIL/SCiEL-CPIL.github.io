"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    mu, sigma, f_best, cost: 1-D arrays of shape (2000,).
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Guard against zero/negative sigma
    sig = np.maximum(sigma, 1e-9)

    # UCB: mu + beta * sigma (we want to minimize, so higher mu is worse)
    # For minimization, we want low mu and high sigma (uncertainty)
    # UCB score = -mu + beta * sigma
    beta = 2.0
    
    ucb = -mu + beta * sig

    # Cost-aware: divide by cost (higher cost = lower priority)
    cost_safe = np.maximum(cost, 1e-9)
    scores = ucb / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses Probability of Improvement (PI) with an exponential cost penalty.
    This strategy strongly prefers cheap points with high probability of beating f_best.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi_prob = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    # Exponential cost penalty
    # The cost is in [1, 2] roughly (1 + mean(x in [0,1])).
    # We want to heavily penalize higher costs.
    # A factor of 5.0 provides a sharp preference for cheaper points as observed in knowledge.
    penalty = np.exp(-5.0 * (cost_safe - 1.0))
    
    score = pi_prob * penalty

    # Clamp non-finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining normalized EI/cost with exploration.
    """
    # Deadline guard: if we're too close to the limit, return safe scores
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)

    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: improvement per unit cost
    ei_score = ei / cost_safe

    # Exploration term: prefer high uncertainty, penalized by cost
    # Using beta=0.4 from the best-performing inspiration
    beta = 0.4
    ucb_score = beta * sigma / cost_safe

    # Normalize each component to zero-mean, unit-variance for comparable scales
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score) + 1e-8
    ei_norm = (ei_score - ei_mean) / ei_std

    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-8
    ucb_norm = (ucb_score - ucb_mean) / ucb_std

    # Combine: primarily exploitation, with meaningful exploration
    # Using 0.75/0.25 split from the best-performing inspiration
    combined = 0.75 * ei_norm + 0.25 * ucb_norm

    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
# EVOLVE-BLOCK-END

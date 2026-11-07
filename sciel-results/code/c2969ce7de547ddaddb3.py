"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Expected Improvement (EI) for minimization, cost-aware
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a Gaussian: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Cost-aware: EI / cost

    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)

    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: higher EI per unit cost is better
    scores = ei / cost

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    # Safety margin check
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END

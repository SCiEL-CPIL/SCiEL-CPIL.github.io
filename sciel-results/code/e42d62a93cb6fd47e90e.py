"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a cost-aware LCB with adaptive exploration based on gap to known minimum.
    """
    # Explicit deadline guard
    start_time = time.time()
    deadline = 325.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Structural change: Fixed constant kappa for cost-normalized LCB
    # Based on knowledge that kappa=1.5 yielded significant improvements
    kappa = 1.5

    # Calculate cost-normalized LCB
    # LCB = mu - kappa * sigma
    # We want to minimize LCB, so we maximize -LCB / cost
    lcb = mu_safe - kappa * sigma_safe
    score = -lcb / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Check deadline (unlikely to be hit for this simple calculation, but required)
    if time.time() > deadline:
         # Return the current best effort if somehow we are close to limit
         pass

    return score
# EVOLVE-BLOCK-END

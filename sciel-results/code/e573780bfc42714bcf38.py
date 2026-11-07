"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    For each candidate, sample a single objective value from the GP posterior
    N(mu, sigma^2) and use the negative sample (lower = better objective)
    divided by cost as the acquisition score. Higher score = evaluate next.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed value so far (lowest since we're minimizing).
    cost : array-like, shape (2000,)
        Cost of evaluating each point, shape (2000,).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    t0 = time.time()
    deadline = 295

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw one sample from each posterior N(mu_i, sigma_i^2)
    # Using standard normal draws scaled by sigma
    z = np.random.standard_normal(n)
    sample = mu + sigma_safe * z

    # For minimization, lower sample = better.
    # Acquisition score: we want higher scores to be selected.
    # Use -sample / cost so that low predicted values (good) give high scores.
    # This implicitly balances exploration (high sigma -> wider sample spread)
    # and exploitation (low mu -> consistently low samples).
    scores = -sample / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

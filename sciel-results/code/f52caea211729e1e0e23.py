"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates using Adaptive LCB.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive LCB for minimization.
    # We want to maximize the score, which corresponds to minimizing the LCB:
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score = -LCB(x) = beta * sigma(x) - mu(x)
    # 
    # Adaptive beta: Explore more when f_best is worse (larger), exploit more when f_best is better (smaller).
    # For Ackley, min is 0. f_best starts around ~5-10 and goes to 0.
    # Let beta decrease as f_best decreases.
    # beta = base + factor * f_best
    # If f_best is 0, beta = base. If f_best is 10, beta = base + 10*factor.
    # This encourages exploration when the current best is far from the optimum.
    base_beta = 1.0
    factor_beta = 0.5
    beta = base_beta + factor_beta * f_best

    # Calculate LCB-based score
    scores = beta * sigma_safe - mu

    # Cost-aware: divide by cost
    scores = scores / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

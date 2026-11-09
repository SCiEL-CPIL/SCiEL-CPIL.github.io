"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = None


def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive Upper Confidence Bound (UCB) per unit cost.
    
    This structural change moves from Expected Improvement (EI) to UCB.
    UCB balances exploitation (-mu) and exploration (beta * sigma).
    For multimodal functions like Griewank, the explicit exploration term
    helps escape local minima more effectively than the implicit exploration
    in EI, especially when the GP is uncertain.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 300s limit, return a simple fallback
    if _get_elapsed_time() > 290:
        # Safe fallback: prioritize low mu (exploitation) with cost awareness
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive UCB Parameter
    # We use a beta that encourages exploration. 
    # In standard BO, beta is often sqrt(2 * log(t)) or similar.
    # Without t, we use a static value tuned for Griewank's multimodality.
    # A value of 1.0-2.0 is typical. Let's try 1.5 for stronger exploration.
    beta = 1.5
    
    # UCB Score for minimization: -mu + beta * sigma
    # We want to maximize this score (higher is better for acquisition).
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: Score per unit cost
    score = ucb_scores / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

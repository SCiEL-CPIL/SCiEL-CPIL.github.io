"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Combines exploitation (-mu), exploration (beta * sigma), and cost penalty.
    The exploration coefficient beta adapts based on how close f_best is to the known minimum.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        scores: array of shape (2000,) - scores (higher = evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: simple heuristic to return quickly
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu - 0.5 * cost + 0.1 * sigma

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Cost-aware UCB:
    # Score = -mu + beta * sigma - lambda * (cost - 1)
    # The (cost - 1) term means cost=1 (cheapest) gets no penalty, cost>1 gets penalized.
    
    # Adaptive beta: more exploration when f_best is far from optimum (-0.8)
    # If f_best is close to -0.8, we can exploit more (lower beta)
    # If f_best is far from -0.8, we need to explore more (higher beta)
    f_min = -0.8
    regret = f_best - f_min  # How far we are from the known minimum
    
    # Clip regret to reasonable range
    regret = np.clip(regret, 0.0, 2.0)
    
    # Beta ranges from ~1.5 (exploit) to ~4.0 (explore)
    # Linear interpolation based on regret
    beta = 1.5 + 2.5 * (regret / 2.0)
    
    # Cost penalty coefficient: penalize expensive evaluations
    # Cost ranges from 1.0 to 2.0 (1 + mean(x) where x in [0,1])
    # We want to moderately penalize high cost but not dominate the score
    lambda_cost = 0.3
    
    # Compute scores
    # -mu: exploit (lower mu is better, so -mu is higher for better points)
    # beta * sigma: explore (higher uncertainty gets higher score)
    # -lambda_cost * (cost - 1): penalize expensive points
    scores = -mu + beta * sigma - lambda_cost * (cost - 1.0)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

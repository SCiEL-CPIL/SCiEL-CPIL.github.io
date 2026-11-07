"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive LCB with quadratic cost penalty.
    
    Structural change from UCB: uses Lower Confidence Bound (LCB) for
    minimization, which provides a different exploration/exploitation
    balance. The exploration term is adaptive based on the median
    uncertainty of the GP, shifting from exploration to exploitation
    as the model becomes more confident.
    
    Score = (f_best - mu + beta * sigma) / cost^2
    
    where beta is adaptively scaled by the median sigma.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margin check
    if time.time() - t0 > 250:
        return np.ones_like(mu) / np.maximum(cost, 1e-6)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive exploration parameter based on median uncertainty
    sigma_med = np.median(sigma_safe)
    if sigma_med < 1e-9:
        sigma_med = 1e-9
    
    # Beta controls exploration strength, scaled by median uncertainty
    # Higher sigma_med -> more exploration (unexplored regions)
    # Lower sigma_med -> less exploration (exploited regions)
    beta = 1.5 * sigma_med
    
    # LCB for minimization: mu - beta * sigma (lower is better)
    # We want higher acquisition scores for points that might be better
    # Score = f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    numerator = f_best - mu + beta * sigma_safe
    
    # Ensure non-negative (points with mu > f_best + beta*sigma get 0)
    numerator = np.maximum(numerator, 0.0)
    
    # Quadratic cost penalty: divide by cost^2
    # This aggressively penalizes expensive points, allowing more
    # evaluations within the fixed budget
    cost_safe = np.maximum(cost, 1e-12)
    cost_penalty = cost_safe ** 2
    
    # Final score
    score = numerator / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > 280:
        # If we are extremely close to the deadline, ensure we return immediately
        pass
    
    return score
# EVOLVE-BLOCK-END

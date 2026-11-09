"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer to track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound acquisition function.
    
    Parameters:
    mu (np.ndarray): Mean of GP predictions, shape (2000,).
    sigma (np.ndarray): Standard deviation of GP predictions, shape (2000,).
    f_best (float): Best observed value so far.
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Return a safe fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # UCB: Exploit + Explore
    # The exploration parameter beta controls the trade-off.
    # For Rastrigin, which has many local minima, a higher beta might help escape local optima.
    # Standard EI uses a fixed 0.01-ish implicit exploration. UCB with beta ~ 1-2 is common.
    beta = 1.5
    
    ucb = mu + beta * sigma
    
    # Cost-aware: divide by cost
    # We want to maximize (improvement) / cost
    # Since we are minimizing, "improvement" is relative to f_best.
    # However, UCB is typically used for maximization. 
    # In minimization, we want to minimize mu - beta*sigma (lower bound) or maximize f_best - (mu - beta*sigma)?
    # Actually, standard UCB for minimization is: LCB = mu - beta * sigma. We want to minimize this.
    # So the score for minimization is -LCB = -mu + beta * sigma.
    # Let's use the minimization version of UCB:
    # Score = (f_best - (mu - beta * sigma)) / cost
    # This rewards points that are likely to be much lower than f_best.
    
    lcb = mu - beta * sigma
    scores = (f_best - lcb) / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

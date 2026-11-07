"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline checking
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe, deterministic fallback (e.g., random or zeros)
        # Since we must return finite scores, return zeros or a constant
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and has a minimum floor to avoid numerical issues
    # Although UCB doesn't divide by sigma, a floor is good practice
    sigma = np.maximum(sigma, 1e-6)
    
    # Cost is given, ensure it's positive
    cost = np.maximum(cost, 1e-6)
    
    # UCB parameter
    # Beta controls exploration. 2.0 is a standard choice.
    beta = 2.0
    
    # UCB Score: mu + beta * sigma
    ucb_scores = mu + beta * sigma
    
    # Cost-aware score: UCB / cost
    # Higher score means we should evaluate this point
    scores = ucb_scores / cost
    
    # Ensure scores are finite
    # If any NaN or Inf, replace with a very low value
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
# EVOLVE-BLOCK-END

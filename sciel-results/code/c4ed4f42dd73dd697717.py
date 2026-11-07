"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300  # Seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Saturated UCB acquisition function with time guard.
    
    Uses tanh saturation on the UCB score to prevent linear over-exploration 
    of high-uncertainty regions, which is critical under tight cost budgets.
    This allows the cost penalty to dominate selection among candidates with 
    'sufficient' expected improvement, improving efficiency.
    """
    # Time guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5:  # 5 second safety margin
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid numerical issues, though GP sigma should be > 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration parameter beta.
    # A value of 2.0 is a standard heuristic for UCB in Bayesian Optimization.
    # It encourages exploration when uncertainty is high.
    beta = 2.0
    
    # UCB Score: (f_best - mu) is the exploitation term (improvement over best).
    # beta * sigma is the exploration term (uncertainty).
    # We maximize this value.
    ucb_score = f_best - mu + beta * sigma
    
    # Apply tanh saturation to the UCB score.
    # This compresses the range of high-improvement candidates, 
    # making the relative difference in cost more significant.
    # The scale factor 1.0 is a heuristic choice.
    saturated_score = np.tanh(ucb_score)
    
    # Cost-aware normalization: Divide by cost to prefer cheaper evaluations.
    # Cost is defined as 1 + mean(x), ranging from 1.0 to 2.0.
    score = saturated_score / cost
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates cost-aware Upper Confidence Bound (UCB) for each candidate.
    
    Strategy:
    1. Use UCB for minimization: Score = -mu + kappa * sigma.
    2. Apply a smooth cost penalty: subtract lambda * cost from the UCB score.
       The cost is 1 + mean(x) for x in [0,1]^3, so it ranges from 1 to 2.
       A small lambda (0.15) provides gentle guidance toward cheaper regions
       without overly biasing the search.
    3. Use a fixed kappa of 1.0, which balances exploration and exploitation 
       effectively for the low budget (~20 evaluations) available.
    4. Include a deadline guard to ensure the function returns within the 
       330s wall-clock limit, although the computation is very fast.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Ensure sigma is positive to avoid division or log issues in other contexts,
    # though here we just multiply.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization
    # We want to minimize f. 
    # Lower mu is better (exploit).
    # Higher sigma is better (explore).
    # Score = -mu + kappa * sigma
    # Higher score means we should evaluate this point next.
    
    kappa = 1.0
    
    ucb_score = -mu + kappa * sigma_safe
    
    # Apply smooth cost penalty
    # Cost ranges from 1 to 2, so normalize by dividing by 2 to get [0.5, 1.0]
    # Then multiply by a small lambda to get a gentle penalty
    lambda_cost = 0.15
    cost_penalty = lambda_cost * cost
    
    final_score = ucb_score - cost_penalty
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Strategy:
    Instead of Expected Improvement (EI), which is non-linear and can be sensitive 
    to the exact value of f_best, we use a UCB approach. UCB balances exploitation 
    (mu) and exploration (sigma) linearly. Dividing by cost ensures we prefer 
    cheaper evaluations.
    
    We use a fixed exploration parameter beta. A value of 2.0-4.0 is typical for 
    early exploration. Since we don't know the iteration count, we use a fixed 
    moderate beta. If we want to adapt, we could use the number of observations, 
    but we don't have that directly. We can infer it from the length of arrays? 
    No, the arrays are always 2000 candidates. We don't know how many points have 
    been evaluated so far.
    
    However, we can look at the 'cost' array. It's just the cost of the candidates.
    
    Let's use a fixed beta = 3.0. This is a strong exploration bonus.
    Score = (mu + beta * sigma) / cost
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to simple EI/cost to ensure fast return
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -30, 30)
        # Import inside to avoid overhead if not needed? No, top level is fine.
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # UCB: mu + beta * sigma
    # We choose beta = 4.0 to encourage exploration in the early stages.
    # As the GP fits better, sigma decreases, and the balance shifts to exploitation.
    beta = 4.0
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb = mu + beta * sigma_safe
    
    # Cost-aware score
    # cost is (1 + mean(x)), so it is always >= 1.
    scores = ucb / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

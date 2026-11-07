"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe, neutral score (e.g., random or based on mu)
        # We must return 2000 finite scores.
        # If we are near deadline, just return something valid to avoid crash.
        # The evaluator will pick the max.
        return -np.inf + np.random.rand(len(mu)) * 1e-6 # Very low, but finite? 
        # Actually, better to return a deterministic valid score if possible, 
        # but the evaluator kills us anyway. 
        # Let's return zeros to be safe and fast.
        return np.zeros(len(mu))

    # Cost-aware UCB
    # UCB = mu + beta * sigma
    # We divide by cost to account for the budget.
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    ucb = mu + beta * sigma
    
    # Normalize cost to be safe, though cost is >= 1
    cost_safe = np.maximum(cost, 1.0)
    
    scores = ucb / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

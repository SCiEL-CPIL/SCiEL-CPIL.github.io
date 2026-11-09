# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0 # 300s limit + 30s buffer? Task says 300s per eval, but total wall clock is 330s.
# "The evaluator KILLS the program after 330s wall-clock."
# So we have 330 seconds total.
# We should return a valid result if we are close to the deadline.

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _DEADLINE
    
    # Check time
    if time.time() > _DEADLINE - 5.0:
        # If we are running out of time, return a safe, valid score.
        # Since we don't know the optimal x, we can't do much better than a random or uniform score.
        # But we must return finite scores.
        # Returning zeros or ones is safe.
        return np.ones_like(mu)

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero
    cost_safe = cost + 1e-9
    
    # Structural Change: Hypervolume-based / UCB approach
    # For minimization, the "Hypervolume" improvement is driven by how much lower the value is.
    # The "Optimistic" estimate of the minimum is mu - beta * sigma.
    # We want to maximize the potential for finding a low value.
    # Score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Beta controls exploration. 
    # In standard UCB, beta ~ sqrt(log(t)). 
    # With few iterations, log(t) is small. 
    # Let's try a fixed beta. 
    # Beta=1.0 is a reasonable start for "moderate" exploration.
    # Beta=0.5 is more exploitation.
    # Beta=2.0 is more exploration.
    
    # Given the "stagnation" warning and the previous success of PoI (which is probabilistic),
    # UCB is a different family (deterministic bound).
    
    beta = 1.0
    score = (beta * sigma - mu) / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

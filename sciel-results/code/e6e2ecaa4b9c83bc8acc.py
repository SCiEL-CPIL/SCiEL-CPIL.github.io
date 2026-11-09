"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t_start = time.time()
    
    # Cost-normalized LCB strategy
    # LCB = mu - beta * sigma (lower bound on the objective)
    # We want to minimize, so lower LCB is better
    # Score should be higher for more promising points
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Use total cost as a proxy for elapsed time/budget usage
    total_cost = np.mean(cost)
    # Beta decays from ~2.0 to ~0.5 as budget is used
    beta = 0.5 + 1.5 * np.exp(-total_cost / 10.0)
    
    # LCB: mu - beta * sigma (we want this to be as low as possible)
    lcb = mu - beta * sigma
    
    # Score: negative LCB divided by cost (higher is better)
    # Since we're minimizing, -lcb/cost rewards low values and low costs
    scores = -lcb / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima early on
    uncertainty_bonus = 0.1 * beta * sigma / cost
    
    scores = scores + uncertainty_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

import numpy as np
from scipy.stats import norm
import time

_start_time = None
_total_budget = 30.0
_initial_cost = 5.0  # 5 uniform random points

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin check
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Ensure numerical stability
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Estimate remaining budget based on current cost
    # cost is the cost of evaluating each candidate, but we need total cost so far
    # From the task: "While total cost is below 30"
    # We don't have direct access to total cost, but we can estimate from the number of evaluations
    # Actually, we need to infer the budget remaining from the inputs
    # Since we don't have the total cost directly, let's use a heuristic
    # The cost array contains evaluation costs, but we need to track total cost
    
    # For now, use a fixed exploration parameter that balances exploitation and exploration
    # kappa scales with how much budget remains
    # Early on: more exploration, later: more exploitation
    
    # Estimate progress: if we assume average cost ~1.5, then after n evaluations, total cost ~1.5*n
    # We can estimate n from the magnitude of f_best or just use a fixed schedule
    
    # Use a dynamic kappa: start high, decrease over time
    # Since we don't have explicit iteration count, use a fixed reasonable value
    # The key insight from the strategy: maximize -(mu + kappa * sigma) / cost
    # which is equivalent to minimizing (mu + kappa * sigma) / cost
    
    # For minimization with cost awareness:
    # We want to find points with low predicted value (mu) and high uncertainty (sigma)
    # per unit cost. The LCB approach: minimize mu + kappa * sigma
    # So we maximize - (mu + kappa * sigma) / cost = (-mu - kappa * sigma) / cost
    
    # Dynamic kappa: larger when we have more budget (early), smaller when less budget (late)
    # Since we can't easily track total cost, use a moderate fixed value
    # Based on the inspiration that combines EI/cost with exploration bonus
    
    # Let's use a weighted combination:
    # score = (-mu - kappa * sigma) / cost
    # with kappa chosen to balance exploration and exploitation
    
    kappa = 2.0
    
    score = (-mu - kappa * sigma) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

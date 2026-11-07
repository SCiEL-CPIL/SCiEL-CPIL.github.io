"""Cost-aware acquisition for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Score = (beta * sigma - mu) / cost
    
    where beta is dynamically tuned based on the number of evaluations
    (inferred from cost). Higher score means more promising to evaluate next.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP standard deviation in objective units
        f_best: lowest observed value
        cost: (2000,) cost of evaluating each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Infer number of evaluations from total cost so far.
    # cost here is per-candidate cost; we don't have total_cost directly,
    # but we can use a heuristic: the mean cost gives a rough scale.
    # Use a dynamic beta that grows slowly with exploration.
    # Since we don't have N_eval directly, use a fixed aggressive beta
    # that balances exploration and exploitation well for rastrigin.
    
    # For rastrigin with its many local minima, we want aggressive exploration.
    # Use beta = 2.0 as a strong exploration factor.
    beta = 2.0
    
    # LCB score: we want to minimize mu - beta*sigma, so maximize beta*sigma - mu
    # Normalize by cost for cost-awareness
    score = (beta * sigma - mu) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

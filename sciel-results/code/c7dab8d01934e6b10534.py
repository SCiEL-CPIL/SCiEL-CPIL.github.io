"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Aware UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Precompute constants
_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit
_BETA = 2.0  # Exploration parameter for UCB

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Strategy:
    1. Compute the optimistic bound for the objective value:
       f_optimistic(x) = mu(x) - beta * sigma(x)
       This represents the lowest value the function might take at x,
       accounting for uncertainty.
       
    2. Compute the potential improvement over the current best:
       I(x) = max(0, f_best - f_optimistic(x))
       
    3. Normalize by the evaluation cost to prioritize cheap improvements.
       Score(x) = I(x) / cost(x)
       
    This approach (UCB) is structurally different from Expected Improvement (EI)
    as it uses a quantile-based confidence bound rather than an expected value
    integration, often leading to more aggressive exploration in uncertain regions.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Deadline guard
    t0 = time.time()
    
    # 1. Compute Optimistic Bound (Minimization)
    # We want to find points where the function could be much lower than f_best.
    # The lower bound of the credible interval is mu - beta * sigma.
    sigma_safe = np.maximum(sigma, _EPS)
    f_optimistic = mu - _BETA * sigma_safe
    
    # 2. Compute Potential Improvement
    # Improvement is positive if the optimistic bound is better than f_best
    improvement = f_best - f_optimistic
    improvement = np.maximum(improvement, 0.0)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_MARGIN:
        cost_safe = np.maximum(cost, _EPS)
        return improvement / cost_safe

    # 3. Cost-awareness
    # Divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, _EPS)
    
    final_score = improvement / cost_safe
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return final_score
        
    return final_score

# EVOLVE-BLOCK-END

"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Global Step-Decay Cost-Adjusted UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_EPS = 1e-12
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

# Global state to track the number of acquisition calls (proxy for steps)
_step_count = 0
# Estimated total steps: (30 - 5) / 1.5 ~ 16.6, let's say 20 to be safe
_EST_TOTAL_STEPS = 20.0
_BETA_MAX = 2.5
_BETA_MIN = 0.5
_COST_PENALTY_EXP = 1.0  # Exponent for cost normalization (1.0 is division)

def acquisition(mu, sigma, f_best, cost):
    """
    Global Step-Decay Cost-Adjusted UCB acquisition function.
    
    Strategy:
    1. Use Upper Confidence Bound (UCB) for exploration/exploitation balance.
    2. Dynamically decay the exploration coefficient (beta) based on a global 
       step counter, approximating the remaining budget.
    3. Normalize the UCB score by the evaluation cost to prefer cheaper candidates 
       with high potential.
    """
    global _step_count
    _step_count += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Calculate dynamic beta based on step count
    # Decay from BETA_MAX to BETA_MIN over EST_TOTAL_STEPS
    progress = min(_step_count / _EST_TOTAL_STEPS, 1.0)
    beta = _BETA_MIN + (_BETA_MAX - _BETA_MIN) * (1.0 - progress)
    
    # Compute UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Cost-adjusted score: UCB / cost
    # This encourages picking points that offer high confidence-adjusted improvement 
    # relative to their cost.
    score = ucb / (cost ** _COST_PENALTY_EXP)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END

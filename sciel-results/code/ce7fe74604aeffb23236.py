"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time at module load
_T0 = time.time()
_DEADLINE = 290.0  # seconds, safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB),
    with cost-awareness via multiplicative discount. The LCB component provides
    exploration pressure that is adaptive to the current best value, while EI
    provides exploitation. The combination balances exploration/exploitation
    more effectively than either alone.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Evaluation costs (in [1, 2]).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    if time.time() - _T0 > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound for minimization:
    # LCB = mu - beta * sigma
    # We want higher scores for lower LCB, so we use -LCB = -mu + beta * sigma
    # The beta parameter controls exploration. We adapt it based on the scale
    # of the problem. A reasonable default is around 1.5-2.0 for 2D problems.
    # We use a slightly lower value to balance with EI.
    beta = 1.5
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both components to comparable scales
    # EI is already in objective units (improvement expected)
    # LCB_score is also in objective units (lower is better, so -LCB is higher)
    
    # Combine EI and LCB with equal weight initially
    # The idea: EI captures expected improvement, LCB captures uncertainty-driven exploration
    combined = 0.5 * ei + 0.5 * lcb_score
    
    # Cost-aware score: multiplicative discount
    # Higher cost -> lower score, prioritizing cheap evaluations
    scores = combined / cost_safe
    
    # Ensure output is finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

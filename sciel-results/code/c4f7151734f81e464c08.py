"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB with cost normalization.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores. Higher scores indicate better candidates.
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # UCB_min = mu(x) - kappa * sigma(x)
    # However, we also need to account for cost.
    # A common approach is to normalize the improvement potential by cost.
    
    # Let's define the "improvement potential" as the distance from the current best.
    # For minimization, a point is promising if mu(x) < f_best.
    # The expected improvement (EI) is a good metric, but UCB is simpler and faster.
    
    # Strategy:
    # Score = (f_best - UCB_min) / cost
    # If UCB_min < f_best, the numerator is positive.
    # If UCB_min > f_best, the numerator is negative, making the score negative.
    # This encourages points that are likely to be better than f_best, adjusted for cost.
    
    # Alternatively, we can use a standard UCB formulation adapted for cost.
    # Let's try: Score = (mu - kappa * sigma) / cost
    # But this doesn't directly compare to f_best.
    
    # Let's stick to a cost-normalized EI-like or UCB-like score.
    # EI is robust. Let's compute EI but divide by cost.
    # EI = E[max(0, f_best - f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # delta = (f_best - mu) / sigma
    # EI = sigma * (delta * Phi(delta) + phi(delta))
    
    # This is O(N) and very fast. Dividing by cost makes it cost-aware.
    # This is structurally different from Pareto front selection which is O(N^2) and binary.
    
    delta = (f_best - mu) / sigma
    
    # To avoid overflow/underflow issues in cdf/pdf for extreme delta values,
    # we can clip delta, although norm.cdf/pdf are generally stable.
    # For very large positive delta (mu << f_best), EI ~ f_best - mu.
    # For very large negative delta (mu >> f_best), EI ~ 0.
    
    # Vectorized computation
    from scipy.stats import norm
    
    cdf_delta = norm.cdf(delta)
    pdf_delta = norm.pdf(delta)
    
    ei = sigma * (delta * cdf_delta + pdf_delta)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize EI per unit cost.
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

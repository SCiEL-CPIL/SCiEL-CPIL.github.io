"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a simple deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Cost-Aware Expected Improvement (EI)
    #
    # The previous approach used a Lower Confidence Bound (LCB), which is a linear
    # combination of the mean and standard deviation. While LCB is effective, it
    # can be overly aggressive in exploration when sigma is large, or overly
    # exploitative when sigma is small, depending on the fixed kappa.
    #
    # Expected Improvement (EI) is a non-linear acquisition function that
    # specifically optimizes for the expected reduction in the objective function
    # value. It has a natural adaptive property:
    # 1. When mu is much lower than f_best (good prediction), EI is high.
    # 2. When sigma is large (high uncertainty), EI increases, encouraging exploration.
    # 3. Crucially, the magnitude of EI reflects the *potential* improvement, not just
    #    the probability. This helps in rugged landscapes where a small probability of
    #    a huge improvement (deep global minimum) might be more valuable than a high
    #    probability of a small improvement (local minimum).
    #
    # Cost Awareness:
    # The task specifies that cost = 1 + mean(x_scaled). This means evaluations in
    # the "center" of the domain are more expensive. To maximize the number of
    # evaluations within the budget (or to minimize regret per unit cost), we
    # normalize the acquisition score by the cost.
    #
    # Score = EI(x) / Cost(x)
    #
    # This effectively asks: "What is the expected improvement per unit of cost?"
    # By dividing by cost, we bias the search towards cheaper points (edges of the
    # domain) if they offer comparable improvement to expensive points (center).
    # However, if a center point offers significantly higher EI, it will still be
    # selected. This is a more principled trade-off than a fixed penalty.
    #
    # Implementation details:
    # 1. Calculate z = (f_best - mu) / sigma
    # 2. EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #    where Phi is the standard normal CDF and phi is the standard normal PDF.
    # 3. Handle numerical stability:
    #    - If sigma is very small, z can be large. norm.cdf and norm.pdf handle this,
    #      but we should clip z to avoid underflow/overflow in extreme cases, although
    #      scipy.stats.norm is generally robust.
    #    - Ensure sigma > 0.
    # 4. Normalize by cost.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # Clip z to prevent extreme values that might cause numerical issues in CDF/PDF
    # Although scipy handles large z well, clipping to [-50, 50] is a safe practice
    # for performance and numerical stability in some edge cases.
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization
    # Cost is at least 1.0. We divide by cost to get "improvement per cost unit".
    cost_safe = np.maximum(cost, 1.0)
    
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

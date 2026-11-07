"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Standard Gaussian Expected Improvement (EI)
    # We use the standard GP EI formula. The previous attempt used Student-t,
    # but the knowledge base suggests that standard EI (or even just EI/cost)
    # might be more effective if the "robustness" of Student-t was causing
    # over-exploration or suboptimal exploitation in this specific setup.
    # Also, the "inspiration" mentioned simple EI/cost.
    
    delta = f_best - mu
    z = delta / sigma_safe
    
    # EI = delta * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Cost-Aware Scaling: Cost-Effective EI (CEEI)
    # The objective is to minimize regret per unit cost.
    # By dividing EI by cost, we prioritize points that offer high expected
    # improvement for low cost. This is a direct structural change from the
    # previous "blended" approach (EI + bonus) / cost^alpha.
    
    # We use a linear cost penalty (alpha=1) rather than a soft power law (alpha<1).
    # This enforces a stricter efficiency constraint, which is often better for
    # budget-constrained BO when the cost function is smooth and predictable.
    
    score = ei / cost_safe
    
    # 3. Numerical Stability
    # Ensure finite values. If EI is negative (which it shouldn't be much due to max(0,.)
    # implicit in the formula structure, though delta can be negative making EI 0),
    # we clip to 0.
    score = np.maximum(score, 0.0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

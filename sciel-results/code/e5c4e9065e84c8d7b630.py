"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 310  # 310s out of 330s budget, safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB (Lower Confidence Bound) for minimization.
    
    Structural change: Instead of EI/UCB (which maximize expected gain/upper bound),
    we minimize the Lower Confidence Bound divided by cost. This directly targets
    regions where the predicted value is low AND uncertainty is high (potential
    minima), while penalizing expensive evaluations. This is a deterministic
    surrogate optimization approach that leverages the structure of the problem.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware mean
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma
    # Lower LCB means potentially better (lower) objective value
    # We want to MINIMIZE LCB, so we negate it for the acquisition score
    # (higher score = better candidate to evaluate)
    
    # Beta controls exploration: higher beta = more exploration of uncertain regions
    # beta = 2.0 provides strong exploration to escape local minima
    beta = 2.0
    
    # LCB score: -LCB = -(mu - beta*sigma) = -mu + beta*sigma
    lcb_score = -mu + beta * sigma
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # This makes the acquisition directly target high-value, low-cost regions
    cost_safe = np.maximum(cost, 1e-9)
    score = lcb_score / cost_safe
    
    # Add a small EI component to ensure exploitation of known good regions
    # This balances exploration (LCB) with exploitation (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Combine: primarily LCB-driven, with a small EI contribution
    # The EI term ensures we don't completely ignore regions near f_best
    score = score + 0.1 * ei / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END

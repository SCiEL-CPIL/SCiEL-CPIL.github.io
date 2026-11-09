# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_t_start = None
_DEADLINE = 310.0

# Pre-compute the z-score for the 10th percentile (lower quantile for minimization)
# We want to minimize the lower quantile of the posterior.
# Q_0.1 = mu + sigma * z_0.1
# z_0.1 is negative.
_Z_LOWER = norm.ppf(0.1)  # approx -1.28155

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()

    elapsed = time.time() - _t_start
    if elapsed > _DEADLINE:
        # Emergency: return a simple valid score
        # Use a simple EI-like score to be safe
        eps = 1e-12
        z = (f_best - mu) / (sigma + eps)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    eps = 1e-12
    
    # Quantile of the Posterior (QoP) for minimization
    # We estimate the 10th percentile of the posterior distribution.
    # Lower quantile = mu + sigma * Z_LOWER (Z_LOWER is negative)
    # We want to find points where this lower quantile is low (better than f_best).
    # Score should be high when (f_best - lower_quantile) is large and positive.
    # lower_quantile = mu + sigma * Z_LOWER
    # f_best - lower_quantile = f_best - mu - sigma * Z_LOWER
    # Since Z_LOWER is negative, -sigma * Z_LOWER is positive.
    # So, f_best - mu + |Z_LOWER| * sigma
    
    improvement = f_best - mu
    exploration = sigma * abs(_Z_LOWER)
    
    # Combine improvement and exploration
    # This is equivalent to maximizing the "optimistic" lower bound
    raw_score = improvement + exploration
    
    # Cost-aware: divide by cost
    # cost is shape (2000,)
    # Ensure cost is positive
    score = raw_score / (cost + eps)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

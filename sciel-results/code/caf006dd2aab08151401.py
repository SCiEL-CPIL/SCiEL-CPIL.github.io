"""Cost-aware acquisition: Adaptive-EI with global uncertainty scaling."""
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # safety margin before the 330 s wall-clock kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Griewank-2D.
    
    Strategy: Expected Improvement (EI) with a non-zero exploration parameter (xi)
    that adapts to the global variance of the GP mean predictions. This provides
    a non-linear exploration bonus that is structurally different from linear UCB
    and more adaptive than fixed-xi EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against degenerate inputs.
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard: fall back to fast EI.
    if time.time() - _START > _LIMIT:
        z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
        s = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
        return np.where(np.isfinite(s), s, 0.0)

    # Calculate Global Exploration Scale
    # Use the standard deviation of the GP mean as a proxy for global uncertainty/spread.
    # If the GP mean is very spread out, we need more exploration.
    global_std_mu = np.std(mu)
    
    # Adaptive xi: scale with global_std_mu.
    # If global_std_mu is large, xi is large.
    # We clip to prevent excessive exploration.
    xi = 0.1 * global_std_mu
    
    # Standard EI formula with xi:
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - xi - mu) / sigma
    f_target = f_best - xi
    z = np.clip((f_target - mu) / sigma, -10.0, 10.0)
    
    ei = (f_target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost.
    scores = ei / cost
    
    # Ensure finiteness.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

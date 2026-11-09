"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track elapsed time to ensure we return well before the 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    # If we're close to the deadline, return a simple UCB score as fallback
    if _check_deadline():
        return mu + 2.0 * sigma

    # Expected Improvement (EI) component
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB (Upper Confidence Bound) component for exploration
    # Use a beta parameter that balances exploration and exploitation
    beta = 2.0
    ucb = mu + beta * sigma_safe

    # Hybrid: blend EI and UCB
    # EI is in objective units (lower is better for minimization, so EI is positive when mu < f_best)
    # UCB is also in objective units
    # We want higher scores to mean "evaluate next"
    # EI gives higher scores where improvement is expected
    # UCB gives lower scores where uncertainty is high (for minimization, we want to explore low mu + high sigma)
    # For minimization, UCB should be mu - beta * sigma (lower is better)
    # But the task says "higher meaning evaluate next"
    # So for minimization:
    #   EI: higher when mu is much lower than f_best (good)
    #   UCB for minimization: we want to explore where mu - beta*sigma is low, so score = -(mu - beta*sigma) = -mu + beta*sigma
    
    # Let's use: score = EI + alpha * (-mu + beta * sigma)
    # This combines exploitation (EI) with exploration (prefer low mu, high sigma)
    
    alpha = 0.5  # weight for UCB exploration term
    score = ei + alpha * (-mu + beta * sigma_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

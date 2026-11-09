"""Cost-aware acquisition for three-hump-camel-2d BO with cost = 1 + mean(x_scaled)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    Uses a modified Expected Improvement with a softened cost penalty.
    The cost is 1 + mean(x_scaled_to_01), ranging from 1 to 2.
    We use a power on cost to soften the penalty (beta < 1) so that
    the search is not overly biased toward cheap (low-x) regions.
    """
    # Time guard: if we've been running too long, return a simple fallback
    # (This is a safety net; the function should be fast enough normally)
    t0 = time.time()

    # Soften the cost penalty: cost^beta with beta < 1
    # beta=0.5 means sqrt(cost), which reduces the relative penalty
    # of more expensive evaluations compared to linear cost
    beta = 0.5
    cost_eff = np.power(cost, beta)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement (standard)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware EI: EI / cost_eff
    # This balances improvement against cost
    scores = ei / cost_eff

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # If we're running out of time, return immediately
    if time.time() - t0 > 280.0:
        return scores

    return scores
# EVOLVE-BLOCK-END

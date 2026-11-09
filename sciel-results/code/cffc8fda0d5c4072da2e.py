"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard: 330s wall-clock limit, use 320s as safety margin
_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple POI/cost
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute Probability of Improvement (POI)
    # POI = P(y(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)

    # Cost-aware score: POI / cost
    # This prioritizes points that are likely to improve the best observed value
    # and penalizes expensive evaluations.
    score = poi / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Add a tiny exploration bonus for points with high uncertainty 
    # but low POI to avoid premature convergence if the GP is miscalibrated.
    # However, the cell finding suggests POI/cost is the key improvement.
    # We will stick to pure POI/cost to align with the successful strategy.
    
    return score
# EVOLVE-BLOCK-END

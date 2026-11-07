"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Knowledge Gradient (KG) approximation.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP standard deviation predictions
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        fallback = sigma_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Knowledge Gradient approximation:
    # KG(x) ≈ E[max(0, f_best - f(x))] but with a correction for
    # the probability that x improves f_best, weighted by the
    # expected improvement magnitude.
    #
    # A common approximation: KG ≈ EI * P(f(x) < f_best)
    # where P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    #
    # This gives more weight to points that are likely to improve
    # the best, while also considering the magnitude of improvement.
    
    diff = f_best - mu
    z = diff / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Standard EI
    ei = diff * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Probability of improvement
    poi = cdf
    
    # KG approximation: EI * POI
    # This emphasizes points that are both likely to improve AND have high expected improvement
    kg_approx = ei * poi
    
    # Add a small exploration bonus for high uncertainty regions
    # to avoid getting stuck in exploitation-only mode
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine KG with exploration
    score_base = kg_approx + exploration_bonus
    
    # Soft cost penalty: cost in [1, 2], use 1 + 0.5*(cost-1) for range [1, 1.5]
    cost_soft = 1.0 + 0.5 * np.clip(cost - 1.0, 0.0, 1.0)
    cost_soft = np.maximum(cost_soft, 1e-6)
    
    score = score_base / cost_soft
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

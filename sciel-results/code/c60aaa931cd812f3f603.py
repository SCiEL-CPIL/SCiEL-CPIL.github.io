"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Knowledge Gradient (KG) for robust optimization.
    
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
    
    # Knowledge Gradient (KG) acquisition function
    # KG approximates the expected reduction in the optimal value
    # For a single observation, KG can be approximated using the normal distribution
    
    # The KG for a single evaluation at x is approximately:
    # E[max(0, f_best - f(x))] weighted by the probability of improvement
    # This is similar to EI but with a different weighting
    
    # A practical approximation of KG:
    # KG ≈ E[max(0, f_best - f(x))] * (1 - Phi((f_best - mu) / sigma))
    # But a simpler and effective approximation is to use:
    # KG ≈ sigma * (phi(z) - z * Phi(-z)) where z = (f_best - mu) / sigma
    
    diff = f_best - mu
    z = diff / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Standard EI
    ei = diff * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # KG approximation: use the expected improvement but weight it differently
    # The key insight is that KG accounts for the fact that observing a value
    # also updates our belief about the global minimum
    # A common approximation is: KG ≈ EI * (1 + 0.5 * sigma / max(|mu - f_best|, 1e-8))
    # This gives more weight to exploration when uncertainty is high relative to distance from best
    
    distance = np.abs(mu - f_best)
    uncertainty_weight = 1.0 + 0.5 * sigma_safe / np.maximum(distance, 1e-8)
    
    kg_approx = ei * uncertainty_weight
    
    # Alternative: use a direct KG-like formula
    # KG ≈ sigma * phi(z) + (f_best - mu) * (1 - Phi(z))
    # This is actually equivalent to EI, so let's use a different approach
    
    # Use a combination: KG emphasizes exploration more than EI
    # Let's use: score = EI + alpha * sigma * phi(z) where alpha controls exploration
    alpha = 0.3
    kg_score = ei + alpha * sigma_safe * pdf
    
    # Soft cost: reduce the penalty for expensive regions
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    # Soft cost: 1 + 0.4 * (cost - 1), giving range [1, 1.4]
    cost_soft = 1.0 + 0.4 * np.clip(cost - 1.0, 0.0, 1.0)
    cost_soft = np.maximum(cost_soft, 1e-6)
    
    score = kg_score / cost_soft
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

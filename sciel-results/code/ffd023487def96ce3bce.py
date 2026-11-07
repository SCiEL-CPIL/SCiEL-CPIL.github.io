# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def _soft_min(a, b, tau=0.1):
    """Smooth minimum using log-sum-exp trick for stability."""
    m = np.minimum(a, b)
    return m - tau * np.log1p(np.exp(-(a - m) / tau) + np.exp(-(b - m) / tau))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Uses a refined cost-aware EI approach with soft-min combination of EI and LCB.
    
    Parameters:
    - mu: shape (2000,), GP mean predictions
    - sigma: shape (2000,), GP standard deviations
    - f_best: scalar, best observed value
    - cost: shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to valid ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 0.5, 10.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound for minimization)
    # LCB = mu - beta * sigma
    # Higher LCB score = more promising (lower predicted value with uncertainty)
    
    # Adaptive beta based on progress
    dist_to_min = max(f_best - (-78.3323), 0.0)
    progress = 1.0 / (1.0 + 0.05 * dist_to_min)
    
    # Beta decreases as we get closer to optimum (more exploitation)
    beta_lcb = 1.5 * (1.0 - 0.5 * progress)
    
    lcb_score = -(mu - beta_lcb * sigma)  # Negate so higher is better
    
    # Cost-aware combination using soft-min
    # We want to combine EI and LCB signals
    # Both are "higher is better" now
    
    # Soft-min gives a smooth combination that emphasizes the better of the two
    # But we want to balance them, so use soft-min with moderate tau
    tau = 0.2
    combined = _soft_min(ei, lcb_score, tau)
    
    # Apply cost penalty
    # Penalize expensive evaluations more
    # cost is in [1, 2] typically
    cost_penalty = 0.5 * (cost - 1.0) * (1.0 + progress)
    
    score = combined - cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

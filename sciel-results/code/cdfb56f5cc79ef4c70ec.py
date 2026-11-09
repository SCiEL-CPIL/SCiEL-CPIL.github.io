"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a robust, risk-averse formulation:
    1. Compute the improvement distribution per point.
    2. Estimate CVaR (Conditional Value-at-Risk) of the improvement, focusing on
       the worst-case 10% of predicted outcomes (robust against GP uncertainty).
    3. Apply a cost penalty that scales with the quantile of the cost distribution,
       discouraging expensive evaluations when the budget is tight.
    """
    t_start = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost normalization: cost is in [1, 2] roughly (1 + mean(x in [0,1]))
    # We'll use a relative cost penalty.
    
    # Compute the improvement: f_best - f(x)
    # We want to maximize E[improvement] or a risk measure of it.
    # Improvement ~ N(f_best - mu, sigma)
    
    # Expected Improvement (baseline)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # CVaR of the improvement distribution:
    # For a normal distribution N(m, s), the CVaR at level alpha (worst alpha fraction)
    # is: CVaR_alpha = m + s * phi(z_alpha) / alpha
    # where z_alpha is the (1-alpha) quantile of the standard normal.
    # Wait, CVaR is the expected value of the worst alpha fraction.
    # For maximization, we care about the worst outcomes of the improvement.
    # Let I = f_best - f. We want to maximize I.
    # The worst 10% of I are the lowest values of I.
    # I ~ N(m, s) where m = f_best - mu, s = sigma.
    # The 10% quantile of I is m + s * z_0.1
    # CVaR_0.1 = E[I | I <= q_0.1] = m + s * phi(z_0.1) / 0.1
    # where z_0.1 = norm.ppf(0.1) ≈ -1.2816
    
    alpha = 0.1
    z_alpha = norm.ppf(alpha)  # negative value
    phi_z_alpha = norm.pdf(z_alpha)
    
    m = f_best - mu
    # CVaR of improvement
    cvar = m + sigma_safe * phi_z_alpha / alpha
    
    # Cost penalty:
    # We want to penalize high cost. The cost is given per point.
    # A simple approach: subtract a term proportional to cost.
    # But we want it to be adaptive. Let's use the rank of the cost.
    # Or simply: score = cvar - lambda * cost
    # What lambda? We can scale it by the typical magnitude of cvar.
    
    # Let's use a hybrid: primarily CVaR, but with a cost-aware adjustment.
    # The cost ranges from 1 to 2. Let's normalize cost to [0, 1] for the penalty.
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # The penalty should be significant enough to influence the choice.
    # Let's set lambda based on the scale of cvar.
    # If cvar is large, we can afford a bit more cost. If cvar is small, we should be very cost-aware.
    # Actually, simpler: just use a fixed lambda that balances the scales.
    # cvar is in objective units (f_best - f). f is around -0.8.
    # Let's try a dynamic lambda: lambda = 0.1 * std(cvar) / (cost_max - cost_min + eps)
    
    cvar_std = np.std(cvar)
    if cvar_std < 1e-9:
        cvar_std = 1e-9
    
    # Heuristic: the cost penalty should be comparable to the variation in cvar.
    # Let's try: score = cvar - 0.5 * cvar_std * cost_norm
    # This way, the cost penalty is scaled by the uncertainty in the improvement.
    
    scores = cvar - 0.5 * cvar_std * cost_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t_start > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

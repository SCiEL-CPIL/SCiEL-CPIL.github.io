"""Rewritten program with a prior-based, cost-aware EI acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # 1. Safety handling for sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # 2. Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using norm.cdf and norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 3. Structural Change: Prior-based Cost Weighting
    # The problem states the minimizer is in the inner 80% of the box.
    # Box [-4, 5] scaled to [0, 1].
    # Inner 80% corresponds to scaled [0.1, 0.9].
    # Cost = 1 + mean(x_scaled).
    # The minimizer's cost follows a distribution centered at 1.5.
    # We weight candidates by how likely their cost is to be near the minimizer's cost.
    # This avoids wasting budget on very cheap (edge) or very expensive (far edge) points
    # where the minimum is known *not* to be.
    
    # Parameters for the cost prior
    cost_mean = 1.5
    # Std dev of the minimizer cost derived from uniform distribution on [0.1, 0.9]
    # Var(X) = (0.8^2)/12 = 0.0533
    # Var(Mean) = 0.0533 / 4 = 0.0133
    # Std(Mean) = sqrt(0.0133) ~ 0.115
    # We use a slightly wider std (0.2) to be less restrictive and allow some exploration
    # if the GP is very confident about a point slightly off-center.
    cost_std = 0.2 
    
    # Gaussian weight for cost
    # We want high weight for cost ~ 1.5, low weight for cost ~ 1.0 or 2.0
    cost_weight = np.exp(-((cost - cost_mean) ** 2) / (2 * cost_std ** 2))
    
    # 4. Combine EI and Cost Weight
    # Score = EI * Cost_Weight
    # This ensures we only explore points that are both promising (high EI) 
    # and located in the region where the minimum is likely to be (high cost_weight).
    
    scores = ei * cost_weight
    
    # 5. Safety: Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # 6. Time Guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        # Fallback to simple EI if taking too long (unlikely for vectorized ops)
        z_fallback = (f_best - mu) / sigma_safe
        ei_fallback = (f_best - mu) * norm.cdf(z_fallback) + sigma_safe * norm.pdf(z_fallback)
        return ei_fallback

    return scores
# EVOLVE-BLOCK-END

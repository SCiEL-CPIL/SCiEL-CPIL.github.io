"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    EI captures the magnitude of expected improvement, which is more informative than PI.
    Normalizing by cost prioritizes evaluations that offer high improvement per unit cost.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma and cost
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0 (likely improvement)
    # If mu > f_best, z < 0 (unlikely improvement)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # For numerical stability, handle large z
    # If z > 8, Phi(z) ~ 1, phi(z) ~ 0 -> EI ~ f_best - mu
    # If z < -8, Phi(z) ~ 0, phi(z) ~ 0 -> EI ~ 0
    
    ei = np.zeros_like(mu)
    
    # Mask for moderate z where we compute full formula
    mask_moderate = np.abs(z) < 8.0
    
    # Compute EI for moderate z
    if np.any(mask_moderate):
        z_mod = z[mask_moderate]
        mu_mod = mu[mask_moderate]
        sigma_mod = sigma_safe[mask_moderate]
        
        phi_z = norm.pdf(z_mod)
        Phi_z = norm.cdf(z_mod)
        
        ei_mod = (f_best - mu_mod) * Phi_z + sigma_mod * phi_z
        ei[mask_moderate] = ei_mod
    
    # For z >= 8 (mu << f_best), EI is approximately f_best - mu
    mask_high = z >= 8.0
    if np.any(mask_high):
        ei[mask_high] = f_best - mu[mask_high]
        
    # For z <= -8 (mu >> f_best), EI is approximately 0
    # ei is already initialized to 0, so no action needed for mask_low
    
    # Clip EI to be non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # Score = EI / cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time (though this operation is fast)
    elapsed = time.time() - start_time
    if elapsed > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END

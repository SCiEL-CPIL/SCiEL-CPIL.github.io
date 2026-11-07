"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining multiplicative cost scaling
    with a small uncertainty-based exploration bonus.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) for minimization.
    2. Apply a soft multiplicative cost factor that gently discounts expensive
       candidates (factor ranges from 1.0 at cost=1 to 0.7 at cost=2).
    3. Add a tiny uncertainty bonus to encourage exploration in high-sigma
       regions, helping escape local optima without overwhelming the EI signal.
    
    This avoids the over-penalization of pure division by cost while still
    preferring cheaper evaluations when gains are comparable.
    """
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Clip small negative values due to floating point errors.
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        # Fallback to simple EI if running out of time
        return ei
    
    # Cost-aware adjustment:
    # Cost is in [1, 2] since it's 1 + mean(x) where x in [0,1]^3
    # Normalize cost to [0, 1] range for the penalty
    cost_clipped = np.clip(cost, 1.0, 2.0)
    cost_norm = (cost_clipped - 1.0) / 1.0  # now in [0, 1]
    
    # Multiplicative factor: starts at 1.0 for cost=1, decreases to 0.7 for cost=2
    # This is softer than division by cost
    cost_factor = 1.0 - 0.3 * cost_norm  # ranges from 1.0 to 0.7
    
    # Small uncertainty bonus for regions with high uncertainty relative to improvement
    # This helps when EI is small but sigma is large, encouraging exploration
    # The ratio sigma / (|f_best - mu| + sigma) is in [0, 1] and peaks when
    # uncertainty dominates the expected improvement
    uncertainty_ratio = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    uncertainty_bonus = 0.1 * uncertainty_ratio
    
    # Combine: base EI scaled by cost factor plus a small uncertainty bonus
    # Scale the uncertainty bonus to be very small relative to typical EI values
    score = ei * cost_factor + uncertainty_bonus * 1e-3
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Final time check
    if time.time() > deadline - 2.0:
        # Fallback to raw EI if running out of time
        score = ei
        score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END

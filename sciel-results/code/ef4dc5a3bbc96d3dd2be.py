import numpy as np
from scipy.stats import norm
import time

def _fast_norm_cdf(x):
    """Fast approximation of standard normal CDF using Abramowitz & Stegun."""
    sign = np.sign(x)
    x = np.abs(x)
    # Use the standard normal CDF for better accuracy
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

def _fast_norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with cost normalization.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - score: array of shape (2000,) with acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    phi_z = _fast_norm_pdf(z)
    Phi_z = _fast_norm_cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    EI = np.maximum(EI, 0.0)
    
    # Cost-normalized EI: divide by cost to favor cheaper evaluations
    # We want higher scores to be evaluated next
    score = EI / safe_cost
    
    # Add a small exploration bonus based on uncertainty
    # This helps when EI is zero (e.g., when mu > f_best)
    # Use a weighted combination: primarily EI, with a small sigma-based bonus
    # The bonus encourages exploring high-uncertainty regions
    
    # Normalize sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = sigma
    
    # Blend: 90% cost-normalized EI + 10% normalized uncertainty
    # This keeps exploitation dominant while allowing some exploration
    score = 0.9 * score + 0.1 * sigma_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

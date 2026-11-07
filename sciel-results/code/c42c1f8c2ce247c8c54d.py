import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    EI is adaptive: it balances exploration (via sigma) and exploitation (via mu)
    based on the current best observed value f_best.
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If f_best is the minimum observed, we want to improve upon it.
    # Improvement = f_best - f(x)
    # E[Improvement] = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    # Using scipy.stats.norm for stability
    
    try:
        phi_z = norm.pdf(z)
        phi_cdf_z = norm.cdf(z)
    except Exception:
        # Fallback if scipy fails, though unlikely
        phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        phi_cdf_z = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        
    # Expected Improvement
    scores = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return scores

    return scores

import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-normalized Expected Improvement (EI).
    
    Score = EI(mu, sigma, f_best) / cost
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # If mu < f_best, Z > 0 (exploitation)
    # If mu > f_best, Z < 0 (exploration)
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    from scipy.stats import norm
    
    EI = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # EI is always non-negative, but let's clamp to be safe
    EI = np.maximum(EI, 0.0)
    
    # Cost penalty: divide by cost
    cost_safe = np.maximum(cost, 1e-12)
    
    score = EI / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        pass
        
    return score

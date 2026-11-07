import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by cost.
    
    Parameters:
    mu: array of shape (N,) - Mean of the GP posterior
    sigma: array of shape (N,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0 - 5.0  # 5 second safety margin
    
    # Ensure finite inputs and correct shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clamp sigma and cost to avoid division issues
    # sigma should be positive. 
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # We clip z to avoid numerical issues with norm.cdf/pdf
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative, z is negative. 
    # Phi(z) will be small, phi(z) will be small. EI should be 0 or close to 0.
    # Actually, the formula assumes we are looking for improvement (lower value).
    # Let delta = f_best - mu.
    # If delta > 0 (mu < f_best), we have improvement.
    # If delta < 0 (mu > f_best), no improvement.
    
    # Standard EI formula:
    # EI = delta * Phi(z) + sigma * phi(z)
    # where z = delta / sigma
    
    delta = f_best - mu
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = delta * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Due to floating point errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost
    score = ei / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return a safe default (zeros) to avoid timeout
        return np.zeros_like(score)
    
    return score

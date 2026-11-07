"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time
_start_time = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Sharpened Probability of Improvement (PI).
    
    This refines the pure PI strategy by applying a scaling factor to the z-score,
    making the acquisition function more aggressive in selecting the most promising
    candidates (exploitation) and reducing the influence of borderline uncertainty
    (exploration noise).
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Time guard: if we're approaching the deadline, return a safe result
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        return np.ones(len(mu)) * 1e-9
    
    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Avoid division by zero in sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Sharpening factor: scales the z-score to make the CDF steeper
    # alpha > 1 increases exploitation (prefers points with mu << f_best more strongly)
    # alpha < 1 increases exploration (softens the decision)
    # Based on the finding that pure PI is effective, we sharpen it slightly to 
    # focus even more on the top candidates.
    alpha = 2.0
    z_sharp = z * alpha
    
    # Probability of Improvement with sharpening
    pi = norm.cdf(z_sharp)
    
    # Normalize by cost
    score = pi / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

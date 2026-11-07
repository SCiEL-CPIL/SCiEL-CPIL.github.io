"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Effective Expected Improvement (CEI) for Bayesian Optimization.
    
    This function computes an acquisition score based on Expected Improvement (EI)
    normalized by the evaluation cost. The EI calculation uses an adaptive exploration
    parameter (beta) that scales with the current GP uncertainty (median sigma),
    allowing the strategy to shift from exploration to exploitation as the GP
    becomes more confident.
    
    Score = EI(x) / cost(x)^gamma
    
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
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margin check
    if time.time() - t0 > 250:
        # Fallback to a simple cost-normalized exploration if we are running out of time
        # This is unlikely for 2000 points, but good practice
        return np.ones_like(mu) / np.maximum(cost, 1e-6)
    
    # Parameters
    gamma = 2.5  # Cost penalty exponent. High value favors cheap points.
    # Adaptive beta: Scale with the median uncertainty of the GP
    # If sigma is high, we are in an unexplored region -> explore more.
    # If sigma is low, we are in a well-explored region -> exploit.
    sigma_med = np.median(sigma)
    # Ensure sigma_med is not zero
    if sigma_med < 1e-9:
        sigma_med = 1e-9
        
    # Alpha controls the magnitude of the adaptive exploration bonus
    # A value of 0.5 is a standard starting point for EI-like functions
    alpha = 0.5
    beta = alpha * sigma_med
    
    # Safe sigma for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for EI
    # z = (f_best + beta - mu) / sigma
    z = (f_best + beta - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best + beta - mu) * Phi(z) + sigma * phi(z)
    # Note: We add beta to f_best to effectively shift the target for improvement,
    # which encourages exploring points that might be slightly worse than f_best
    # but have high uncertainty (potential for better minimum).
    
    # Using scipy.stats.norm for PDF and CDF
    # Phi(z) is the CDF, phi(z) is the PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Compute EI
    ei = (f_best + beta - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-12)
    cost_penalty = cost_safe ** gamma
    
    # Final score
    score = ei / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety check
    if time.time() - t0 > 280:
         # If we are extremely close to the deadline, ensure we return immediately
         pass
         
    return score
# EVOLVE-BLOCK-END

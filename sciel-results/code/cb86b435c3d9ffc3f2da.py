"""Cost-aware acquisition for three-hump-camel 2D BO using Cost-Adjusted EI."""
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lower is better)
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)

    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI: score = EI / cost
    # This directly models the improvement per unit cost
    cost_safe = np.maximum(cost, 1e-10)
    score = ei / cost_safe
    
    # Add a small exploration bonus for high uncertainty
    # This helps when EI is very small (near convergence) but uncertainty remains
    sigma_max = np.max(sigma) + 1e-10
    sigma_norm = sigma / sigma_max
    
    # Scale exploration bonus to be small relative to typical EI values
    # Use a fraction of the median EI to keep it in the right scale
    ei_median = np.median(ei) if np.any(ei > 0) else 1e-6
    exploration_bonus = 0.02 * sigma_norm * ei_median
    
    score += exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        # Return whatever we have, it's already computed
        pass
    
    return score

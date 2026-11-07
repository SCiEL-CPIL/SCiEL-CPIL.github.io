import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian expected utility with soft-thresholding.
    
    Uses a bounded probability-like signal based on soft-thresholded improvement,
    divided by evaluation cost to create a cost-efficiency score.
    """
    # Deadline guard: if we've been running too long, return a simple fallback
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:  # Safety margin before 330s limit
        # Fallback: simple cost-scaled LCB
        sigma_safe = np.maximum(sigma, 1e-9)
        beta = 1.0
        scores = (beta * sigma_safe - mu) / cost
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute improvement: how much better than f_best we expect
    improvement = np.maximum(0, f_best - mu)
    
    # Soft-thresholding transform: converts continuous improvement to bounded [0, 1]
    # lambda controls how quickly the signal saturates
    lambda_param = 1.0 / max(f_best, 1e-6)  # Scale relative to current best
    
    # Bounded probability-like signal
    # 1 - exp(-lambda * improvement) gives a smooth, bounded signal
    bounded_signal = 1.0 - np.exp(-lambda_param * improvement)
    
    # Add uncertainty component: regions with high sigma get extra credit
    # This helps exploration in uncertain regions
    uncertainty_bonus = 0.1 * sigma_safe / max(f_best, 1e-6)
    
    # Combine bounded signal with uncertainty
    utility = bounded_signal + uncertainty_bonus
    
    # Cost-efficiency: divide by cost to prefer cheaper evaluations
    # Cost ranges from 1.0 to 2.0
    cost_efficiency = utility / cost
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(cost_efficiency), cost_efficiency, -np.inf)
    
    return scores

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a structural
    exploration bonus for high-uncertainty, promising-mean candidates.
    
    The primary signal is EI (exploitation of known improvements).
    A structural bonus is added to candidates that are in the top 5% of
    uncertainty (sigma) AND have a predicted mean better than the current best.
    This decouples exploration from the raw magnitude of variance, focusing
    instead on relative uncertainty rank among candidates that are still
    plausible to be optimal.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero for division
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # Z = (f_best - mu) / sigma
    # If mu < f_best, Z > 0 (potential improvement)
    # If mu > f_best, Z < 0 (likely no improvement)
    z = (f_best - mu) / sigma_safe
    
    # Clip Z to prevent numerical overflow/underflow in norm functions
    # norm.cdf(-10) is ~ 0, norm.cdf(10) is ~ 1
    # norm.pdf is small for |z| > 4
    z = np.clip(z, -10, 10)
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * P(f < f_best) + sigma * phi(Z)
    # P(f < f_best) = Phi(Z) where Z = (f_best - mu) / sigma
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 2. Compute Quantile-based Exploration Bonus
    # We want to identify candidates that are:
    # a) In the top 5% of uncertainty (high sigma)
    # b) Have a predicted mean that is better (lower) than f_best
    
    # Calculate the 95th percentile of sigma
    sigma_threshold = np.percentile(sigma, 95)
    
    # Mask for high uncertainty
    high_uncertainty_mask = sigma >= sigma_threshold
    
    # Mask for promising mean (mu < f_best)
    promising_mean_mask = mu < f_best
    
    # Combine masks: only consider points that are both high uncertainty AND promising
    exploration_mask = high_uncertainty_mask & promising_mean_mask
    
    # Define the bonus value
    # The bonus should be significant enough to compete with EI but not dominate it entirely.
    # Scale the bonus by the potential improvement range to be adaptive.
    range_improvement = f_best - np.min(mu)
    if range_improvement < 1e-6:
        range_improvement = 1.0 # Avoid division by zero or tiny values
        
    # A bonus of 10% of the potential improvement range
    bonus_value = 0.1 * range_improvement
    
    # Apply bonus to eligible candidates
    scores = ei.copy()
    scores[exploration_mask] += bonus_value
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores

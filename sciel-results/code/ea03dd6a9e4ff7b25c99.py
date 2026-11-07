import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    quantile-based exploration bonus.
    
    The score is primarily driven by EI (exploitation of known improvements)
    but includes a structural switch: if a candidate is in the top 5% of
    uncertainty (sigma) among the batch AND its predicted mean is better than
    the current best, it receives a bonus. This encourages exploring high-uncertainty
    regions that are still plausible to be optimal, rather than just the
    highest variance regions (which might be far from the optimum).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero for division
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip Z to prevent numerical overflow/underflow in norm functions
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
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
    
    # Scale bonus by the potential improvement range
    range_improvement = f_best - np.min(mu)
    if range_improvement < 1e-6:
        range_improvement = 1.0
        
    bonus_value = 0.1 * range_improvement
    
    scores = ei.copy()
    scores[exploration_mask] += bonus_value
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores

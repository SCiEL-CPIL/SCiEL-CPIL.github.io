import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Quantile-Adaptive Probability of Improvement (PoI) for minimization.
    
    Strategy:
    Uses the Probability of Improvement (PoI) as the core acquisition metric,
    which estimates the probability that a candidate will have a lower objective
    value than the current best (f_best).
    
    To address the potential over-exploration of PoI in early stages and 
    under-exploration in later stages, we introduce a dynamic scaling factor 
    based on the quantiles of the sigma distribution. This helps in maintaining 
    a balance between exploring high-uncertainty regions and exploiting 
    low-mean regions.
    
    The score is computed as:
    score = Phi((f_best - mu) / sigma) + adaptive_weight * sigma
    
    Where Phi is the CDF of the standard normal distribution, and 
    adaptive_weight is derived from the quantiles of sigma.
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Calculate z-scores for Probability of Improvement
    # We want P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Compute CDF of standard normal
    # Using scipy.stats.norm.cdf would be ideal, but we can use erf from math or scipy.special
    # Since SciPy is available, we use it for accuracy
    from scipy.special import ndtr
    poi_scores = ndtr(z)
    
    # Ensure poi_scores are finite
    poi_scores = np.nan_to_num(poi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Calculate adaptive weight based on sigma quantiles
    # This helps in scaling the exploration term dynamically
    sigma_q25 = np.percentile(sigma, 25)
    sigma_q75 = np.percentile(sigma, 75)
    sigma_range = sigma_q75 - sigma_q25
    
    if sigma_range > 1e-9:
        # Normalize sigma to [0, 1] range based on IQR
        normalized_sigma = (sigma - sigma_q25) / sigma_range
        normalized_sigma = np.clip(normalized_sigma, 0, 1)
        # Adaptive weight: higher for high uncertainty regions
        adaptive_weight = 0.1 * normalized_sigma
    else:
        adaptive_weight = np.zeros_like(sigma)
    
    # Final score: PoI + adaptive exploration bonus
    scores = poi_scores + adaptive_weight * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return best effort (though this shouldn't happen for such simple computation)
        return scores
        
    return scores

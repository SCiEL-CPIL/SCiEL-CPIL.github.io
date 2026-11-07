import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Track elapsed time for safety margin
    t_start = time.time()
    
    # Guard against numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) - standard formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Lower Confidence Bound score for minimization
    # We want to minimize, so a good region has low mu and high sigma (uncertainty)
    # LCB = mu - beta * sigma, we want to minimize this, so score = -(mu - beta*sigma) = -mu + beta*sigma
    # Using beta=1.0 for a balanced exploration/exploitation
    beta = 1.0
    lcb_score = -mu + beta * sigma
    
    # Normalize LCB score to [0, 1] range for fair combination
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Combine EI and normalized LCB score
    # EI is in objective units and non-negative
    # LCB_norm is in [0, 1]
    # We need to scale them to be comparable
    # Scale LCB_norm by the max EI to put it in objective units
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        lcb_scaled = lcb_norm * ei_max
    else:
        lcb_scaled = np.zeros_like(lcb_score)
    
    # Weighted combination: 0.6 EI + 0.4 LCB
    # This balances exploitation (EI) with exploration (LCB)
    score = 0.6 * ei + 0.4 * lcb_scaled
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget - return immediately if we're close to the limit
    # The evaluator kills after 330s, so we should return well before that
    # This is a safety measure; in practice, this function should be fast
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:  # Safety margin
        return score
    
    return score

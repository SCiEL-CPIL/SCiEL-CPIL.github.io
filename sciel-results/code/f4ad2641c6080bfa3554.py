import numpy as np
def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration weight scaled by f_best
    # High f_best (early) -> High beta (explore)
    # Low f_best (late) -> Low beta (exploit)
    beta = f_best
    
    # Quantile-normalized standard deviation
    sigma_med = np.median(sigma)
    if sigma_med < 1e-9:
        sigma_med = 1e-9
        
    sigma_norm = sigma / sigma_med
    
    # Thompson Sampling heuristic: -mu + beta * sigma
    # This encourages exploring high variance regions early
    score = -mu + beta * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

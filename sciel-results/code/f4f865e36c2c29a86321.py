import numpy as np

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration weight with smooth transition
    # Use a smooth function that transitions from exploration to exploitation
    # f_best starts high (~1-10 for Ackley) and approaches 0
    # We want beta to be large when f_best is large, small when f_best is small
    # Use: beta = f_best / (1 + f_best) * max_beta + min_beta
    # This gives beta in [min_beta, max_beta) with smooth transition
    # At f_best=0: beta = min_beta (pure exploitation)
    # At f_best->inf: beta -> max_beta (max exploration)
    # For Ackley, f_best starts around 1-10, so let's use max_beta=3, min_beta=0.1
    max_beta = 3.0
    min_beta = 0.1
    beta = min_beta + (max_beta - min_beta) * f_best / (1.0 + f_best)
    
    # Quantile-normalized standard deviation for robustness
    # Use 75th percentile as a more conservative scale estimate
    sigma_scale = np.percentile(sigma, 75)
    if sigma_scale < 1e-9:
        sigma_scale = 1e-9
        
    sigma_norm = sigma / sigma_scale
    
    # Clip to prevent extreme outliers from dominating the score
    sigma_norm = np.clip(sigma_norm, 0, 5)
    
    # Thompson Sampling heuristic: -mu + beta * sigma_norm
    # This encourages exploring high variance regions early
    score = -mu + beta * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

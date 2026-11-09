import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Quantile-normalize sigma: divide by the median sigma of the batch
    # This makes the exploration term relative to the current model's uncertainty level
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
        
    sigma_norm = sigma_safe / med_sigma
    
    # Dynamic exploration weight scaled by f_best
    # Aggressive early (large f_best), focused late (small f_best)
    # We use f_best directly as the weight.
    # If f_best is ~3, weight is 3. If f_best is ~0.1, weight is 0.1.
    # This ensures the exploration term is comparable in magnitude to the exploitation term (-mu)
    # when the model is uncertain about the best point.
    lambda_ts = f_best
    
    # Thompson Sampling heuristic score
    # We want to maximize this score.
    # -mu favors low predicted values (exploitation)
    # lambda * sigma_norm favors high uncertainty (exploration)
    score = -mu + lambda_ts * sigma_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

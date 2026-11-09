import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Adaptive Lower Confidence Bound
    # Increase exploration (beta) when the current best is poor (high f_best)
    # and decrease it when the current best is good (low f_best).
    
    # Clip f_best to avoid extreme values affecting beta too much
    f_best_clipped = np.clip(f_best, 0.0, 25.0)
    
    # Beta starts at 2.0 and increases with the square root of the regret
    beta = 2.0 + 1.5 * np.sqrt(f_best_clipped)
    
    # LCB Score: beta * sigma - mu
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

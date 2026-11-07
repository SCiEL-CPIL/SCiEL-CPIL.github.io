import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: Maximize (kappa * sigma - mu)
    # This is equivalent to minimizing (mu - kappa * sigma), the Lower Confidence Bound.
    kappa = 1.5
    sigma_safe = np.maximum(sigma, 1e-9)
    scores = kappa * sigma_safe - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize f(x).
    # The lower confidence bound on the function value is mu - beta * sigma.
    # We want to minimize this, so we maximize the negative: -mu + beta * sigma.
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Adaptive beta
    # f_best starts high (up to 25) and decreases to 0.
    # We use a beta that starts higher (more exploration) and decreases (more exploitation).
    # Base beta of 0.5 to ensure some exploration even at the end.
    # Scaling factor based on f_best relative to the max possible value (25).
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = 0.5 + 1.0 * np.sqrt(ratio)
    
    # Score = -mu + beta * sigma
    # Higher score means more likely to be the minimum.
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END

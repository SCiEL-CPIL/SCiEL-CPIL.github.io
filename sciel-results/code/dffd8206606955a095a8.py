import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 325.0  # 5s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Structural change: Use a deterministic valley-following heuristic.
    
    The Beale function has a characteristic valley structure. 
    Instead of a blended probabilistic score (EI/UCB), we use a 
    composite score that explicitly rewards low GP mean (exploiting known low regions)
    while using a dynamic gamma to balance against uncertainty.
    
    score = -mu + gamma * sigma
    
    gamma is dynamically increased based on the ratio of f_best to the 
    observed minimum of mu, and also based on the "spread" of sigma.
    This allows the function to shift from exploration (high gamma) to 
    exploitation (low gamma) as the search converges.
    """
    # Check deadline
    if time.time() - _START_TIME > TIME_LIMIT:
        # Fallback: return deterministic scores based on mu only (exploitation)
        mu = np.asarray(mu)
        return -mu

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return -mu
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Heuristic for dynamic gamma
    # 1. Look at the ratio of f_best to the minimum mu.
    #    If f_best is much larger than min(mu), we are far from optimum -> explore more.
    #    If f_best is close to min(mu), we are near optimum -> exploit more.
    mu_min = np.min(mu)
    
    # If f_best is very close to mu_min, gamma should be small (exploit)
    # If f_best is much larger than mu_min, gamma should be larger (explore)
    
    # Scale factor: how much does f_best exceed the best predicted value?
    # Use a robust scale based on the spread of mu
    mu_spread = np.max(mu) - mu_min
    if mu_spread < 1e-9:
        mu_spread = 1.0
    
    # Ratio: 0 if f_best == mu_min, 1+ if f_best >> mu_min
    ratio = max(0.0, (f_best - mu_min) / mu_spread)
    
    # 2. Also consider the overall uncertainty level
    #    If sigma is high everywhere, we are in an unexplored region -> explore more
    #    If sigma is low everywhere, we have good information -> exploit more
    sigma_mean = np.mean(sigma_safe)
    
    # Combine both signals
    # gamma starts at a base value and increases with ratio and sigma_mean
    # Base gamma: 0.1 (strong exploitation)
    # Max gamma: 1.0 (strong exploration)
    
    # Normalize ratio to [0, 1] using a soft cap
    ratio_norm = 1.0 - np.exp(-ratio)
    
    # Normalize sigma_mean to [0, 1] using a soft cap
    # Assume typical sigma in [0, 1] for standardized GP
    sigma_norm = 1.0 - np.exp(-sigma_mean)
    
    # Dynamic gamma: blend of the two signals
    gamma = 0.1 + 0.9 * (0.5 * ratio_norm + 0.5 * sigma_norm)
    
    # Compute the deterministic score
    # score = -mu + gamma * sigma
    scores = -mu + gamma * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

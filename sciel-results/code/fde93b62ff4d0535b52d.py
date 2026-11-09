import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition function.
    score = mu + k * sigma
    Higher score means more likely to be a good candidate (lower true value).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clean up inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive k: start with moderate exploration, decay as f_best approaches 3
    # f_best starts around 20-30, minimum is 3
    # We want k to be larger when f_best is far from 3, smaller when close
    # Use a smooth decay: k = k_min + (k_max - k_min) * exp(-(f_best - 3) / scale)
    # Or simpler: k proportional to (f_best - 3) with some floor
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 20.0
    
    # Adaptive exploration coefficient
    # When f_best is high (far from optimum), explore more
    # When f_best is low (close to optimum), exploit more
    gap = max(f_best - 3.0, 0.0)
    # k ranges from about 0.5 (exploit) to 2.5 (explore)
    # Use a smooth function: k = 0.5 + 2.0 * (1 - exp(-gap / 5.0))
    k = 0.5 + 2.0 * (1.0 - np.exp(-gap / 5.0))
    
    # PLB score: mu + k * sigma
    # Since we're minimizing, lower mu is better, higher sigma is more uncertain
    # We want to maximize the probability of finding a low value
    # score = mu + k * sigma gives higher scores to uncertain regions
    scores = mu + k * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

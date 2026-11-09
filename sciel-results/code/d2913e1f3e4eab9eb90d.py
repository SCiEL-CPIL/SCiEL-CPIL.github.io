import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a normalized hybrid of Expected Improvement
    and Upper Confidence Bound (UCB) for minimization.
    """
    t_start = time.time()
    deadline = t_start + 280.0
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to maximize this score.
    # Low mu is good, high sigma is good.
    # Score = -mu + beta * sigma
    # We choose beta = 1.0 initially, but normalization will handle scaling.
    # However, a fixed beta=1.0 might be too small or large.
    # Let's use a heuristic beta based on the scale of sigma.
    # Or just use beta=1.0 and let normalization do the work.
    ucb = -mu + 1.0 * sigma
    
    # 3. Normalize both scores to have mean 0 and std 1
    # This makes the combination robust to the absolute scale of mu, sigma, and f_best.
    def normalize_scores(s):
        s_mean = np.mean(s)
        s_std = np.std(s)
        if s_std < 1e-8:
            return np.zeros_like(s)
        return (s - s_mean) / s_std
        
    ei_norm = normalize_scores(ei)
    ucb_norm = normalize_scores(ucb)
    
    # 4. Combine
    # Equal weight hybrid
    scores = ei_norm + ucb_norm
    
    # Check time budget
    if time.time() > deadline:
        pass # Just return what we have
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

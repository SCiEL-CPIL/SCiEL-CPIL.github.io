import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 330.0  # 300s limit + margin, but we should be much faster
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_scores = norm.cdf(z)
    
    # To avoid saturation where many points have PI ~ 1.0 or ~ 0.5,
    # we add a small exploitation term. This ensures that among points
    # with similar probability of improvement, the one with the lower
    # predicted mean is preferred. This prevents the optimizer from
    # getting stuck in high-variance regions where PI is moderate
    # but the mean is actually worse than f_best.
    # The coefficient 1e-4 is small enough to not override the PI signal
    # significantly, but large enough to break ties and guide exploitation.
    exploitation_bonus = 1e-4 * (f_best - mu)
    
    scores = pi_scores + exploitation_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores

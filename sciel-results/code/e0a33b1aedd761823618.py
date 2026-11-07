import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Structural change: Probability of Improvement (PI) acquisition function
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This models the task as binary classification of "improvement"
    # providing a robust, bounded score in [0, 1] that prioritizes
    # high-confidence improvements over exploratory gambles
    
    delta = f_best - mu
    z = delta / sigma
    
    # PI is the probability that the function value at x is less than f_best
    pi_score = norm.cdf(z)
    
    # The PI score is already bounded in [0, 1] and naturally balances
    # exploration and exploitation:
    # - Points with mu well below f_best get high PI (exploitation)
    # - Points with high sigma near f_best get moderate PI (exploration)
    # - Points with mu well above f_best get low PI (avoided)
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(pi_score), pi_score, 0.0)
        return scores
    
    # Ensure finite output
    scores = np.where(np.isfinite(pi_score), pi_score, 0.0)
    
    return scores

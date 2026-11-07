import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    This is a structural change from the previous UCB+EI hybrid.
    PI focuses on the probability that a candidate point improves upon the current best.
    It is more exploitative than EI and does not include an explicit uncertainty bonus
    term like UCB, relying instead on the GP's posterior distribution.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Calculate z-score for each candidate
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to prevent numerical overflow/underflow in CDF
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    pi_score = norm.cdf(z)
    
    # Ensure finite output
    score = np.where(np.isfinite(pi_score), pi_score, 0.0)
    
    return score

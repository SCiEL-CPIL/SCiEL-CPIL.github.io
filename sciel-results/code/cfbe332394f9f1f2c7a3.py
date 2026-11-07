import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Quantile-based Exploration Strategy.
    
    This strategy combines the Probability of Improvement (PI) with a 
    bonus based on the empirical rank of the uncertainty (sigma).
    It prioritizes points that are likely to improve upon f_best 
    (high PI) and among those, favors those with higher uncertainty 
    (high sigma rank), effectively balancing exploitation and exploration 
    without relying on a fixed magnitude scaling for sigma.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best
    p_imp = norm.cdf(z)
    
    # Calculate the rank of sigma among all candidates
    # argsort gives indices that would sort sigma
    # We want the percentile rank: 0.0 for lowest sigma, 1.0 for highest
    # Using np.argsort(np.argsort(sigma)) gives ranks 0..N-1
    ranks = np.argsort(np.argsort(sigma_safe))
    # Normalize ranks to [0, 1]
    # If N=1, rank is 0, which is fine.
    n = len(sigma_safe)
    if n > 1:
        sigma_ranks = ranks / (n - 1)
    else:
        sigma_ranks = np.array([0.0])
        
    # Weight the probability of improvement by the uncertainty rank
    # alpha controls the strength of the exploration bonus
    # alpha=0 corresponds to Probability of Improvement (PI)
    # alpha=1 or 2 adds significant weight to uncertainty
    alpha = 2.0
    
    score = p_imp * (1.0 + alpha * sigma_ranks)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

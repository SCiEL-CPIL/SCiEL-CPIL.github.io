"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative and handle division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement (PI)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # Since we are minimizing, improvement is f(x) < f_best
    # The GP posterior is N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi_score = norm.cdf(z)
    
    # Exploration Bonus:
    # The prompt mentions that an exploration bonus encouraging moderate uncertainty improved scores.
    # We add a term proportional to sigma, but normalize it to prevent large sigma values
    # (which occur early in the search) from dominating the score entirely.
    # Normalizing by the max sigma in the batch makes the bonus scale-invariant relative to the current batch.
    max_sigma = np.max(sigma)
    if max_sigma > 1e-9:
        normalized_sigma = sigma / max_sigma
    else:
        normalized_sigma = np.zeros_like(sigma)
        
    # Balance between PI (exploitation) and Uncertainty (exploration)
    # Lambda controls the exploration weight. 
    # A value around 0.5-1.0 is typical for balancing.
    # We use 1.0 to give significant weight to exploration, as suggested by the "measured" findings.
    lambda_explore = 1.0
    
    scores = pi_score + lambda_explore * normalized_sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

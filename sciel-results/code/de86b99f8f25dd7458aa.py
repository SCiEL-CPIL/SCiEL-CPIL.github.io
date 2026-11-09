import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Probability of Improvement (PI) divided by cost.
    # PI is more robust than EI in high-dimensional multimodal landscapes
    # as it focuses on the likelihood of improvement rather than the magnitude.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better
    # We penalize expensive points by dividing by cost
    scores = pi / cost
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

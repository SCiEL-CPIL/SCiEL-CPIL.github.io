import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # For minimization, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) term for minimization
    # We want to maximize the potential improvement.
    # Optimistic value: f_best - mu + beta * sigma
    # This encourages points with low mean and high uncertainty.
    beta = 1.5
    ucb = f_best - mu + beta * sigma
    
    # Combine EI and UCB
    # EI is more conservative, UCB is more exploratory.
    # A blend allows for a balance.
    score = 0.5 * ei + 0.5 * ucb
    
    # Cost-aware penalty
    # Divide by cost to prefer cheaper evaluations
    scores = score / cost
    
    # Ensure outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

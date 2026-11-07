import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a blend of Expected Improvement (EI) and UCB, normalized by cost.
    The EI component provides a probabilistic measure of improvement, while
    UCB adds an exploration bonus. Cost normalization biases the search
    toward cheaper regions (lower mean coordinates).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB: f_best - (mu - beta * sigma)
    # Using a moderate beta for exploration
    beta = 1.5
    ucb = f_best - (mu - beta * sigma_safe)
    
    # Blend EI and UCB
    # Weighting: 0.6 * EI + 0.4 * UCB
    # This leverages UCB's stronger exploration signal while incorporating
    # EI's probabilistic dampening in regions where mu is near f_best
    blend = 0.6 * ei + 0.4 * ucb
    
    # Cost-awareness: normalize by cost
    # Cost is in [1, 2], so dividing by cost penalizes expensive points
    scores = blend / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

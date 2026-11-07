import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for EI
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is (f_best - y).
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment
    # We want to maximize EI per unit cost.
    # However, simple division by cost can be unstable if cost is very small.
    # The prompt specifies cost = 1 + mean(x_scaled).
    # x_scaled is in [0, 1]. So cost is in [1, 2].
    # Dividing by cost is safe.
    
    # To make it more robust, we can use a small epsilon in the denominator
    # or simply divide. Given cost >= 1, direct division is stable.
    scores = ei / cost
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

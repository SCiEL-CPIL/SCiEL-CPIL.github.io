import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and an Upper Confidence Bound (UCB) exploration term.
    
    Args:
    mu (np.ndarray): Mean of the GP posterior, shape (N,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,).
    f_best (float): Best observed function value.
    cost (np.ndarray): Cost of evaluating each candidate, shape (N,).
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: Beta * Sigma
    # Beta is tuned to balance exploration and exploitation.
    # A higher beta encourages exploring high-uncertainty regions.
    beta = 2.0
    exploration = beta * sigma_safe
    
    # Total benefit: EI + Exploration
    benefit = ei + exploration
    
    # Cost-aware score: Benefit / Cost
    # Cost is at least 1.0, so division is safe.
    score = benefit / cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

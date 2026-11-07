import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) 
    and a UCB-style exploration bonus.
    
    Parameters:
    mu: Array of mean predictions (shape 2000,)
    sigma: Array of standard deviations (shape 2000,)
    f_best: Current best observed value (scalar)
    cost: Array of costs for each candidate (shape 2000,)
    
    Returns:
    scores: Array of scores (shape 2000,), higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Prevent division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a UCB-style exploration term: beta * sigma
    # This encourages exploration of high-uncertainty regions,
    # which is important when the minimum location is unknown (shifted instances).
    beta = 1.5
    ucb_bonus = beta * sigma_safe
    
    # Combine EI and exploration bonus
    # We want to maximize this score.
    # Normalize by cost to prefer cheaper points if the benefit is similar.
    # Cost is at least 1.0, so division is safe.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (ei + ucb_bonus) / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

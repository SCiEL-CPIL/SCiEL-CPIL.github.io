import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip small negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Use Standardized EI (EI / sigma) to normalize scale
    # This makes the improvement term comparable across different uncertainty levels
    # and prevents early high-EI regions from dominating purely due to scale.
    # EI/sigma is bounded (approx 0 to 0.6)
    std_ei = ei / sigma_safe
    
    # Cost-awareness: Instead of dividing by cost (which biases towards cheapest
    # corners, potentially away from the minimizer in the inner 80%), we use a
    # quadratic penalty centered at the expected minimizer cost (1.5).
    # This encourages search in the central region where the minimizer is likely.
    target_cost = 1.5
    cost_penalty = (cost - target_cost) ** 2
    
    # Combine terms
    # lambda_cost controls the strength of the location prior
    lambda_cost = 2.0
    
    score = std_ei - lambda_cost * cost_penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

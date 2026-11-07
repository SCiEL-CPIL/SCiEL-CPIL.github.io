import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function with global budget-aware scaling.
    
    Parameters:
    mu (np.array): GP mean for 2000 candidates, shape (2000,)
    sigma (np.array): GP standard deviation for 2000 candidates, shape (2000,)
    f_best (float): Best observed function value
    cost (np.array): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.array: Scores for the 2000 candidates, higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute mean cost of the candidate batch
    mean_cost = np.mean(cost_safe)
    
    # Global budget-aware scaling: subtract penalty proportional to cost relative to mean cost
    # This shifts exploration-exploitation balance based on remaining budget
    cost_penalty = 0.5 * (cost_safe / mean_cost - 1.0)
    
    # Normalize EI by its scale to make penalty comparable
    ei_scale = np.std(ei) + 1e-9
    normalized_ei = ei / ei_scale
    
    # Combine normalized EI with cost penalty
    score = normalized_ei - cost_penalty
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

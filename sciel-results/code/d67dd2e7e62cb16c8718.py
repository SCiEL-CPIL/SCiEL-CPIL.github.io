import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    cost (np.array): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.array: Acquisition scores, shape (2000,), higher is better.
    """
    # Ensure inputs are float64 arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle numerical stability for sigma
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of the standard normal distribution
    # Using scipy.stats is robust for 2000 points
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted EI with exploration bonus
    # Score = (EI + exploration_bonus) / cost
    # Exploration bonus encourages visiting low-cost regions
    # The cost is 1 + mean(x_scaled), so lower cost means closer to center
    # We add a small bonus proportional to (1/cost) to favor cheap evaluations
    cost_safe = np.maximum(cost, 1e-9)
    
    # Add a small exploration component that favors low cost
    # This helps escape local optima by exploring cheap regions
    exploration_weight = 0.1
    exploration_bonus = exploration_weight / cost_safe
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

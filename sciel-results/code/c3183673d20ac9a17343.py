import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound) 
    adapted for minimization and cost.
    
    Structural Change: Switched from Expected Improvement (EI) to UCB.
    EI is greedy and can get stuck in local minima in higher dimensions 
    with sparse initial samples. UCB provides a more robust balance 
    between exploration (high sigma) and exploitation (low mu).
    """
    # Ensure sigma is non-negative and avoids division issues later if needed
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to find x that minimizes f(x).
    # The lower confidence bound is LCB(x) = mu(x) - beta * sigma(x).
    # We want to maximize the potential for improvement, which corresponds to 
    # minimizing LCB. Thus, we maximize (beta * sigma(x) - mu(x)).
    
    # Beta parameter:
    # In 4D with ~5 initial points, uncertainty is high.
    # A beta of 2.0 encourages strong exploration initially.
    # As sigma decreases near observed points, the term beta*sigma shrinks,
    # and the algorithm naturally shifts to exploitation (lower mu).
    beta = 2.0
    
    # Calculate the raw UCB score (higher is better)
    ucb_raw = beta * sigma - mu
    
    # Apply cost penalty
    # The problem states cost = 1 + mean(x_scaled), which is between 1 and 2.
    # We divide by cost to prioritize cheaper evaluations.
    # This is a heuristic for "efficiency".
    scores = ucb_raw / cost
    
    # Ensure all scores are finite
    # If cost is 0 (shouldn't happen as min cost is 1), or mu/sigma are inf, handle it.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

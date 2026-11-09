import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware UCB Acquisition Function.
    
    Score = (Beta * Sigma + Max(0, F_best - Mu)) / Cost
    
    This prioritizes points with high uncertainty (Sigma) or high potential improvement 
    (F_best - Mu), penalized by the cost of evaluation.
    """
    # Ensure sigma is positive to avoid numerical issues, though GP sigma should be > 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration term: Reward uncertainty
    beta = 2.0
    explore = beta * sigma
    
    # Exploitation term: Reward predicted improvement over current best
    # Only count if predicted mean is better than current best
    exploit = np.maximum(0.0, f_best - mu)
    
    # Combine and divide by cost
    # Cost is shape (N,)
    scores = (explore + exploit) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

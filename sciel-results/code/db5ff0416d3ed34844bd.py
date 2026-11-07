import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is safe for division
    cost_safe = np.maximum(cost, 1e-10)
    
    # Cost-Effective UCB (Upper Confidence Bound) for minimization
    # We maximize this score.
    # Exploitation term: f_best - mu (positive if mu < f_best)
    # Exploration term: kappa * sigma
    # Penalty: / cost
    
    kappa = 1.5
    
    # Calculate the UCB score
    # Note: If mu is NaN or Inf, the result will be NaN/Inf.
    # We handle finite checks at the end.
    scores = (f_best - mu + kappa * sigma) / cost_safe
    
    # Ensure scores are finite. 
    # If mu or sigma are invalid, score becomes invalid.
    # We replace non-finite values with 0 (or a very low number) to avoid selecting them.
    # However, if all are invalid, we might have issues, but the problem guarantees valid GP outputs usually.
    # Replacing with 0 is safe as we maximize, and 0 is likely lower than valid positive scores.
    # If all scores are 0, we pick index 0.
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

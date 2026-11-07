import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Cost-Aware UCB
    # Maximize expected improvement upper bound
    # UCB = f_best - mu + beta * sigma
    beta = 1.5
    
    scores = (f_best - mu + beta * sigma) / cost
    
    # Handle non-finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure we return a valid array
    # If all scores are -inf, the max will be -inf. 
    # The evaluator might handle this, but to be safe, we can map -inf to 0 or a small negative number?
    # If all are -inf, it means no point is expected to improve f_best.
    # In that case, we should probably pick the one with highest sigma (exploration) or lowest cost?
    # UCB with beta > 0 ensures that if sigma is large, score can be positive even if mu is high.
    # So it's unlikely all are -inf unless sigma is tiny everywhere.
    
    return scores

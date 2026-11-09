import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses an Upper Confidence Bound (UCB) strategy where the exploration parameter beta
    adapts to the current best observed value (f_best). This allows for aggressive
    exploration early in the optimization (when f_best is large) and precise
    exploitation later (when f_best is small).
    
    The score is defined as S = -mu + beta * sigma, which we maximize to select the next point.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: scale exploration based on how far we are from the optimum
    # If f_best is large, we explore more. If f_best is small, we exploit more.
    # Using sqrt(f_best) provides a sub-linear increase in exploration as f_best grows.
    beta = max(0.5, np.sqrt(f_best))
    
    # Calculate UCB score for minimization: we want to maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Handle numerical artifacts
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score

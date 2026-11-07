import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score indicates a better candidate for evaluation.
    
    # Adaptive beta: balance exploration vs exploitation
    # In early stages (high uncertainty), explore more
    # As we get more evaluations, beta could decrease slightly
    # For 30 evaluations total, use a moderate fixed beta
    
    beta = 1.5
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute UCB scores
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize to avoid extreme values
    score_min = np.min(scores)
    score_max = np.max(scores)
    score_range = score_max - score_min
    
    if score_range > 1e-12:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones_like(scores) * 0.5
    
    return scores

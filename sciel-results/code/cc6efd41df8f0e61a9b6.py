import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We want to maximize this score.
    
    # Fixed beta for exploration/exploitation trade-off.
    # 2.0 is a reasonable default for small budgets.
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate UCB scores
    scores = -mu + beta * sigma_safe
    
    # Handle any potential numerical issues (though mu/sigma should be finite)
    # Replace non-finite scores with a very low value to ensure they are not selected
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores

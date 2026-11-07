import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) for minimization with an adaptive beta.
    
    The UCB acquisition function balances exploitation (low mean) and exploration (high uncertainty).
    For minimization, we want to maximize:
    Score(x) = -mu(x) + beta * sigma(x)
    
    We use an adaptive beta based on the spread of the posterior means to scale the exploration term appropriately.
    This addresses the scaling issues of fixed-beta UCB in 1D optimization.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate the standard deviation of the posterior means to determine the scale of the problem
    # This helps in setting an appropriate exploration weight relative to the prediction uncertainty
    mu_std = np.std(mu)
    
    # Avoid division by zero or extremely small beta if mu is constant
    if mu_std < 1e-6:
        mu_std = 1.0
        
    # Adaptive beta: scaled by the spread of the means
    # 0.5 is a heuristic balance factor, but structurally it's driven by data
    beta = 0.5 * mu_std
    
    # UCB for minimization: we want low mu and high sigma
    # So we maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e18)
    
    return score

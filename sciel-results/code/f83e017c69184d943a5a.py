import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Array of shape (N,) with posterior means.
        sigma: Array of shape (N,) with posterior standard deviations.
        f_best: Scalar, the best observed function value so far.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher scores indicate
        more promising candidates.
    """
    # UCB for minimization: We want to maximize -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    # If mu or sigma contain inf/nan, replace them
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return scores

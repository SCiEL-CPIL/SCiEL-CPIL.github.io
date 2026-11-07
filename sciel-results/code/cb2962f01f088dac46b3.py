import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates, higher is better.
    """
    # UCB formula for minimization:
    # We want to select points with low predicted value (mu) and high uncertainty (sigma).
    # Since the output should be "higher is better", we invert the sign of mu.
    # score = -mu + kappa * sigma
    
    kappa = 1.0
    
    # Handle potential numerical issues with sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

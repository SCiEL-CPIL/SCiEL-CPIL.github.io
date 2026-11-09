import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Clipped Thompson Sampling acquisition function.
    
    We sample from the GP posterior y ~ N(mu, sigma^2).
    Since the Beale function is known to be non-negative (minimum 0),
    we clip the sampled values to be at least 0. This prevents the
    algorithm from being overly attracted to regions where the GP
    predicts negative values (which are impossible for the true objective),
    thereby enforcing the known bounds of the objective function and
    focusing exploration on promising, physically valid regions.
    
    Args:
        mu: array of shape (N,) - GP mean
        sigma: array of shape (N,) - GP standard deviation
        f_best: float - best observed value so far
        
    Returns:
        array of shape (N,) - scores, higher is better
    """
    # Ensure sigma is positive to avoid division issues or zero variance issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Draw samples from the posterior
    # y_i ~ N(mu_i, sigma_i^2)
    eps = np.random.standard_normal(mu.shape)
    samples = mu + sigma_safe * eps
    
    # Enforce the known constraint: f(x) >= 0
    # This acts as a prior that the objective cannot be negative.
    samples_clipped = np.maximum(samples, 0.0)
    
    # We want to minimize the objective, so we maximize the score.
    # Score = -objective
    return -samples_clipped

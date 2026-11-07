import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Generate random samples from the posterior distribution
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We draw a sample for each candidate point
    xi = np.random.randn(mu.shape[0])
    
    # Compute the sampled objective values
    # f_sample = mu + sigma * xi
    f_samples = mu + sigma * xi
    
    # We want to minimize the objective, so we maximize the negative sampled value
    scores = -f_samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

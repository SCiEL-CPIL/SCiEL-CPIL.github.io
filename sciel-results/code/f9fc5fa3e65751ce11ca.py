import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: np.ndarray of shape (2000,), predicted mean.
        sigma: np.ndarray of shape (2000,), predicted standard deviation.
        f_best: float, best observed value so far.
        
    Returns:
        score: np.ndarray of shape (2000,), scores where higher is better.
    """
    # Ensure sigma is positive to avoid issues with zero variance
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate a standard normal random vector
    # This samples one function from the Gaussian Process posterior
    z = np.random.randn(len(mu))
    
    # Compute the sampled objective values
    y = mu + sigma * z
    
    # We are minimizing the objective function.
    # Therefore, we want to select the candidate with the lowest sampled value.
    # To fit the "higher score is better" interface, we return the negative of the sampled value.
    score = -y
    
    return score

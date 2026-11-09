import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    return scores

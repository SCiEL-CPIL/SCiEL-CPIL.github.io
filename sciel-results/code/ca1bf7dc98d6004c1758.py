import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function scores.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior at candidate points, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior at candidate points, shape (2000,).
    f_best (float): The lowest observed value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher scores indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized improvement Z
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute the Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # norm.pdf(z) gives phi(z)
    # norm.cdf(z) gives Phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

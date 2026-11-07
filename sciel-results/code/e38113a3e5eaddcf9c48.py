import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement and exploration.
    
    This approach combines the exploitation signal from EI with a controlled
    exploration term to balance finding new minima versus refining known ones.
    The exploration term is scaled by the uncertainty (sigma) to encourage
    visiting regions where the model is less certain.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of the standard normal distribution
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement for Gaussian distribution
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Add an exploration term based on uncertainty
    # This encourages visiting regions with high uncertainty
    # The coefficient is tuned to balance exploration vs exploitation
    exploration_coeff = 0.1
    exploration = exploration_coeff * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure finite scores to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

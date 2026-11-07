import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement + Exploration Term.
    
    This function computes a score for each candidate point based on the
    Gaussian Process posterior. It combines Expected Improvement (EI)
    with a linear exploration term proportional to the standard deviation
    (sigma) to balance exploitation and exploration.
    
    Parameters:
    mu: np.ndarray, shape (N,)
        Predicted mean of the objective function.
    sigma: np.ndarray, shape (N,)
        Predicted standard deviation of the objective function.
    f_best: float
        The best (lowest) objective value observed so far.
        
    Returns:
    np.ndarray, shape (N,)
        Acquisition scores. Higher scores indicate more promising points.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to a small positive value to avoid division by zero
    # and numerical instability.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Exploration Term
    # We add a term proportional to sigma to encourage exploring regions
    # with high uncertainty, even if the current mean estimate is not
    # exceptionally low. This mimics a UCB (Upper Confidence Bound)
    # behavior added to EI.
    beta = 1.0
    exploration = beta * sigma_safe
    
    # Total Score
    scores = ei + exploration
    
    # Ensure all scores are finite
    # Replace any NaN or Inf with a very low value
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores

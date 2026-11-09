import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization with a small exploration bonus.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps in early stages when f_best is not yet close to the optimum
    # The bonus is scaled by the relative uncertainty
    exploration_bonus = 0.1 * sigma_safe
    
    # Final score: higher is better
    score = ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

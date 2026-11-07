import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PI) with dynamic threshold xi.
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior std devs
    f_best: float, best observed value so far
    Returns: (N,) array of scores (higher = better)
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic improvement threshold xi
    # Scale linearly with f_best to target significant relative improvements
    # When f_best is large (early), xi is large -> more exploration
    # When f_best is small (late), xi is small -> more exploitation
    xi = 0.1 * f_best
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    z = (f_best - xi - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

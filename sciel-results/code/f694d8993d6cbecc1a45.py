import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) with constant uncertainty boost.
    
    PI directly maximizes P(f(x) < f_best), which is well-suited for
    finding new global minima in multimodal landscapes like Branin.
    A small constant times sigma prevents premature convergence where
    PI becomes zero in regions with low predicted improvement but
    high uncertainty.
    """
    # PI: P(mu - sigma*z < f_best) = P(z > (mu - f_best)/sigma)
    #     = norm.cdf((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    pi_scores = norm.cdf(z)
    
    # Small constant uncertainty boost to maintain exploration
    # in multimodal regions where PI saturates to 0 or 1
    uncertainty_boost = 0.1 * sigma
    
    scores = pi_scores + uncertainty_boost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

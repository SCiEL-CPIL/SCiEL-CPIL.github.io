import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Branin 2D.
    
    KG estimates the expected reduction in the best observed value
    after evaluating a candidate point. This provides a principled
    exploration-exploitation balance that accounts for the probability
    that a candidate becomes the new global minimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient for minimization:
    # KG(x) = E[min(f(x), f_best) - min(f(x), f_best) | f(x) < f_best]
    #       = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #         - (f_best - mu) * (1 - Phi((f_best - mu)/sigma))
    # Simplifying: KG(x) = (f_best - mu) * (2*Phi(z) - 1) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF of the standard normal
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # KG = (f_best - mu) * (2*Phi(z) - 1) + sigma * phi(z)
    # This can be rewritten as:
    # KG = (f_best - mu) * Phi(z) - (f_best - mu) * (1 - Phi(z)) + sigma * phi(z)
    #    = (f_best - mu) * Phi(z) - (f_best - mu) * Phi(-z) + sigma * phi(z)
    
    kg = (f_best - mu) * (2.0 * phi_z - 1.0) + sigma_safe * pdf_z
    
    # Ensure non-negative scores (KG should theoretically be >= 0)
    kg = np.maximum(kg, 0.0)
    
    # Add a small exploration bonus weighted by uncertainty to help navigate
    # multi-modal landscapes. This encourages exploration of high-uncertainty
    # regions even when KG is low.
    exploration_bonus = 0.05 * sigma_safe
    
    scores = kg + exploration_bonus
    
    return scores

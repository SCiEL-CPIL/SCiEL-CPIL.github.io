import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: Array of shape (N,) with posterior means.
    sigma: Array of shape (N,) with posterior standard deviations.
    f_best: Scalar, the best observed value so far.
    
    Returns:
    scores: Array of shape (N,) with EI scores. Higher is better.
    """
    # Avoid division by zero in sigma
    # If sigma is 0, the improvement is deterministic: max(0, f_best - mu)
    # However, in EI formula, if sigma -> 0:
    # z = (f_best - mu) / sigma
    # If f_best < mu, z -> -inf, Phi(z) -> 0, phi(z) -> 0, EI -> 0
    # If f_best > mu, z -> +inf, Phi(z) -> 1, phi(z) -> 0, EI -> f_best - mu
    # So we can handle sigma=0 safely by masking or just letting the math work 
    # if we avoid explicit division by zero where possible, or use safe sigma.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This formula is for minimization where we want to find values lower than f_best.
    # Improvement I = max(0, f_best - Y)
    # E[I] = (f_best - mu) * P(Y < f_best) + sigma * phi(z)
    # P(Y < f_best) = Phi((f_best - mu) / sigma) = Phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

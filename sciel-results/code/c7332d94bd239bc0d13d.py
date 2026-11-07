import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    # Z = (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best
    # Improvement = f_best - f(x)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative.
    # If mu is much larger than f_best, z is very negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    # If mu is close to f_best, z ~ 0, EI ~ sigma * phi(0).
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is always non-negative in theory, but numerical errors might make it tiny negative?
    # Actually, the formula is positive for all z.
    # However, if sigma was 0, we set it to 1.0, so z is huge if f_best != mu.
    # If f_best == mu, z=0, EI = sigma * phi(0). But if sigma was 0, uncertainty is 0, so EI should be 0 if mu >= f_best.
    # Let's handle sigma == 0 explicitly.
    zero_sigma_mask = sigma < 1e-10
    if np.any(zero_sigma_mask):
        # If sigma is 0, the prediction is deterministic.
        # Improvement is 0 if mu >= f_best.
        # If mu < f_best, we already have a better point? No, f_best is the min observed.
        # If mu < f_best, it implies the model thinks there's a point better than observed?
        # But if sigma=0, the model is certain.
        # Usually, EI is 0 if the mean is worse than best.
        ei[zero_sigma_mask] = np.where(mu[zero_sigma_mask] < f_best, f_best - mu[zero_sigma_mask], 0.0)
        
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

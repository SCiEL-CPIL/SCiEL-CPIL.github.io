import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        EI scores for each candidate, shape (N,). Higher is better.
    """
    # Avoid division by zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # EI formula for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values (though EI should be finite if inputs are)
    # If sigma is 0, EI is max(0, f_best - mu). 
    # The formula handles sigma->0 correctly: z->inf if f_best>mu, z->-inf if f_best<mu.
    # norm.cdf(inf)=1, norm.pdf(inf)=0 => EI = f_best - mu.
    # norm.cdf(-inf)=0, norm.pdf(-inf)=0 => EI = 0.
    # This is correct behavior for zero uncertainty.
    
    return ei

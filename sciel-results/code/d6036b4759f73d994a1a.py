import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu : numpy array, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : numpy array, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    scores : numpy array, shape (N,)
        EI values for each candidate. Higher values indicate better candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero or log issues
    # Although GP sigma should be > 0, numerical errors can occur.
    sigma = np.maximum(sigma, 1e-10)
    
    # Improvement is defined as f_best - f(x), where f(x) ~ N(mu, sigma^2).
    # We want to maximize the expected improvement.
    
    # Z-score of the best value relative to the posterior
    # z = (f_best - mu) / sigma
    # If f_best < mu, z is negative, meaning the current point is likely better than best.
    # If f_best > mu, z is positive, meaning the current point is likely worse.
    
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    
    # Calculate PDF and CDF
    # phi(z) = pdf of standard normal
    # Phi(z) = cdf of standard normal
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # EI can be negative if mu > f_best + sigma (very unlikely to improve), but theoretically max(0, EI)
    # However, standard EI is defined as E[max(f_best - f(x), 0)].
    # The formula above calculates E[f_best - f(x)] * P(f(x) < f_best) + ... 
    # Actually, the closed form E[max(A, 0)] for A ~ N(m, s^2) is m*Phi(m/s) + s*phi(m/s).
    # Here m = f_best - mu.
    # So the formula is correct and yields non-negative values.
    
    # Handle any potential NaNs or infinities
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e9, neginf=0.0)
    
    return scores

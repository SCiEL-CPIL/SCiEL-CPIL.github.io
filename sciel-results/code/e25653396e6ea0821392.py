import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function using Truncated Gaussian Sampling.
    
    We sample from the GP posterior y ~ N(mu, sigma^2), constrained to y >= 0.
    Since the Beale function is non-negative, we use the truncated normal distribution
    to draw samples. This is more accurate than simple clipping as it preserves
    the posterior probability mass correctly for the feasible region.
    
    Args:
        mu: array of shape (N,) - GP mean
        sigma: array of shape (N,) - GP standard deviation
        f_best: float - best observed value so far (not used in pure TS)
        
    Returns:
        array of shape (N,) - scores, higher is better (negative of sampled value)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate alpha = -mu / sigma
    # If sigma is very small, alpha can be huge.
    alpha = -mu / sigma_safe
    
    # Calculate the CDF of the standard normal at alpha
    # P(Z < alpha)
    cdf_alpha = norm.cdf(alpha)
    
    # To avoid numerical issues where cdf_alpha is exactly 1.0 (for large alpha)
    # or 0.0 (for large negative alpha), we clip it slightly away from boundaries
    # if we were using uniform sampling directly, but here we use the transformation.
    # The term (1 - cdf_alpha) can be very small.
    
    # Draw uniform random numbers
    u = np.random.random(mu.shape)
    
    # Calculate the quantile in the standard normal distribution
    # q = u * (1 - cdf_alpha) + cdf_alpha
    # We need to be careful if cdf_alpha is 1.0.
    # If cdf_alpha == 1.0, then 1 - cdf_alpha = 0. q = 1.0. ppf(1.0) is inf.
    # This happens when mu is very negative relative to sigma.
    # In that case, the truncated normal is extremely concentrated near 0.
    # The sample should be very small positive.
    
    # Let's handle the case where cdf_alpha is close to 1.
    # If cdf_alpha is 1, the distribution is degenerate at 0?
    # Actually, if mu -> -inf, the truncated normal mass is at 0.
    
    # A more stable way:
    # If alpha is large (mu << 0), the sample is approximately 0.
    # If alpha is small/negative (mu >= 0), it's standard normal.
    
    # Let's use the direct formula but clip the input to ppf to avoid inf/nan
    # ppf(1.0) is inf. We want large finite values.
    # However, if the sample is inf, score is -inf, which is bad.
    
    # Let's clip the quantile input to [1e-10, 1 - 1e-10]
    quantile_input = u * (1.0 - cdf_alpha) + cdf_alpha
    quantile_input = np.clip(quantile_input, 1e-10, 1.0 - 1e-10)
    
    z_samples = norm.ppf(quantile_input)
    
    samples = mu + sigma_safe * z_samples
    
    # Due to numerical precision, samples might still be slightly negative
    samples = np.maximum(samples, 0.0)
    
    return -samples

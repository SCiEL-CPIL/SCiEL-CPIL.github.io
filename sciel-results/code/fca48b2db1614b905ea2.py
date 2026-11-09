import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid strategy: Probability of Improvement (PI) for exploration
    when the current best is not yet good, and Expected Improvement (EI)
    for exploitation when the current best is reasonably good.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # Higher is better (more likely to improve)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # Higher is better (larger expected reduction in objective)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Adaptive strategy:
    # Hartmann 6D global minimum is approx -3.32237.
    # If f_best is still high (e.g., > -1.5), we are likely in the exploration phase.
    # PI is more robust for exploration as it doesn't penalize high uncertainty as much as EI might in some contexts,
    # and it directly measures the chance of finding a better point.
    # If f_best is lower, we switch to EI to refine the search around the best found region.
    
    # Threshold tuned for Hartmann 6D landscape
    threshold = -1.5
    
    if f_best > threshold:
        return pi
    else:
        return ei
